# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:42:11 2020

@author: yuxi1
"""

from tensorflow.keras import Model, datasets
import tensorflow as tf
from utils import split_dataset,linear_rampup,mixmatch,interleave,semi_loss,ema,weight_decay
from model import WideResNet
import numpy as np
from tqdm import tqdm

#%%parameter
learningrate = 0.002
batchsize = 64
epoch = 1024
num_label = 4000
num_validation = 5000
lambda_u_max = 100
#%%

(x_train, y_train),(x_test, y_test) = datasets.cifar10.load_data()
#x_train = normalize_image(tf.cast(x_train, dtype=tf.float32))
y_train = tf.one_hot(y_train, depth=10, dtype=tf.float32)
y_train = tf.squeeze(y_train,axis=1)
x_train = x_train/255
x_train = x_train*2-1
cifar10_train_dataset = tf.data.Dataset.from_tensor_slices({
        'image': x_train,
        'label': y_train
    })

y_test = tf.one_hot(y_test, depth=10, dtype=tf.float32)
y_test = tf.squeeze(y_test,axis=1)
x_test = x_test/255
x_test = x_test*2-1
cifar10_test_dataset = tf.data.Dataset.from_tensor_slices({
        'image': x_test,
        'label': y_test
    })

trainX, trainU, validation = split_dataset(cifar10_train_dataset, 4000, 5000,10)
#%%
model = WideResNet(10, depth=28, width=2)
model.build(input_shape=(None, 32, 32, 3))
optimizer = tf.keras.optimizers.Adam(lr=0.01)
# model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
# manager = tf.train.CheckpointManager(model_ckpt, f'{ckpt_dir}/model', max_to_keep=3)

ema_model = WideResNet(10, depth=28, width=2)
ema_model.build(input_shape=(None, 32, 32, 3))
ema_model.set_weights(model.get_weights())
# ema_ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=ema_model)
# ema_manager = tf.train.CheckpointManager(ema_ckpt, f'{ckpt_dir}/ema', max_to_keep=3)

#%%
def train(trainX, trainU, model, ema_model, optimizer, epoch):
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=64, drop_remainder=True)

    iteratorX = iter(shuffle_and_batch(trainX))
    iteratorU = iter(shuffle_and_batch(trainU))

    progress_bar = tqdm(range(1024), unit='batch')
    for batch_num in progress_bar:
        lambda_u = 100 * linear_rampup(epoch + batch_num/1024, 16)
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffle_and_batch(trainX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(trainU))
            batchU = next(iteratorU)

        #args['beta'].assign(np.random.beta(args['alpha'], args['alpha']))
        beta = np.random.beta(0.75,0.75)
        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, batchX['image'], batchX['label'], batchU['image'], 0.5, 2, beta)
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, 64)
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)

            # compute loss
            xe_loss, l2u_loss = semi_loss(XUy[:64], logits_x, XUy[64:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss

        # compute gradients and run optimizer step
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, ema_model, 0.999)
        weight_decay(model=model, decay_rate=0.02 * 0.01)

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32), model(tf.cast(batchX['image'], dtype=tf.float32), training=False))

        progress_bar.set_postfix({
            'XE Loss': f'{xe_loss_avg.result():.4f}',
            'L2U Loss': f'{l2u_loss_avg.result():.4f}',
            'WeightU': f'{lambda_u:.3f}',
            'Total Loss': f'{total_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    return xe_loss_avg, l2u_loss_avg, total_loss_avg, accuracy


def validate(dataset, model, epoch,split):

    accuracy = tf.keras.metrics.Accuracy()
    xe_avg = tf.keras.metrics.Mean()
    dataset = dataset.batch(64)
    for batch in dataset:
        logits = model(batch['image'], training=False)
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch['label'], logits=logits)
        xe_avg(xe_loss)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy(prediction, tf.argmax(batch['label'], axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} XE Loss: {xe_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')
    return xe_avg, accuracy    
#%%
import time
for epoch in range(2): 
    start_time = time.time()
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=64, drop_remainder=True)

    iteratorX = iter(shuffle_and_batch(trainX))
    iteratorU = iter(shuffle_and_batch(trainU))

    progress_bar = tqdm(range(1024), unit='batch')
    for batch_num in progress_bar:
        lambda_u = 100 * linear_rampup(epoch + batch_num/1024, 16)
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffle_and_batch(trainX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(trainU))
            batchU = next(iteratorU)

        #args['beta'].assign(np.random.beta(args['alpha'], args['alpha']))
        beta = np.random.beta(0.75,0.75)
        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, batchX['image'], batchX['label'], batchU['image'], 0.5, 2, beta)
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, 64)
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)

            # compute loss
            xe_loss, l2u_loss = semi_loss(XUy[:64], logits_x, XUy[64:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss

        # compute gradients and run optimizer step
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, ema_model, 0.999)
        weight_decay(model=model, decay_rate=0.02 * 0.002)

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32), model(tf.cast(batchX['image'], dtype=tf.float32), training=False))

        # progress_bar.set_postfix({
        #     'XE Loss': f'{xe_loss_avg.result():.4f}',
        #     'L2U Loss': f'{l2u_loss_avg.result():.4f}',
        #     'WeightU': f'{lambda_u:.3f}',
        #     'Total Loss': f'{total_loss_avg.result():.4f}',
        #     'Accuracy': f'{accuracy.result():.3%}'
        # })
                if batch_num % 512 == 0:
            print("Step: {}, xe: {:.3f},l2u:{:.4},total: {:.3f},w_u:{:.2f},acc: {:.2%},time:{:.2f}s"
                  .format(batch_num + 1,xe_loss_avg.result(),l2u_loss_avg.result(),
                 total_loss_avg.result(),lambda_u,accuracy.result(),(time.time() - start_time)))
    xe_avg, test_accuracy = validate(cifar10_test_dataset, ema_model, epoch,'test')
    with open('./log/test.txt','a') as f:
        f.write("Step: {}, xe: {:.3f},l2u:{:.4},total: {:.3f},w_u:{:.2f},acc: {:.2%},test_acc:{:.2%}\n"
                      .format(batch_num + 1,xe_loss_avg.result(),l2u_loss_avg.result(),
                     total_loss_avg.result(),lambda_u,accuracy.result(),test_accuracy.result()))
    print("Epoch: {}, xe: {:.3f},l2u:{:.3f},w_u: {:.3f},total: {:.3f},acc: {:.2%},time:{:.2f}s"
      .format(epoch + 1,xe_loss_avg.result(),l2u_loss_avg.result(),
      lambda_u,total_loss_avg.result(),accuracy.result(),(time.time() - start_time)))

#%%
for epoch in range(100, 200):
    xe_loss, l2u_loss, total_loss, accuracy = train(trainX, trainU, model, ema_model, optimizer, epoch)
    val_xe_loss, val_accuracy = validate(validation, ema_model, epoch, split='Validation')
    test_xe_loss, test_accuracy = validate(cifar10_test_dataset, ema_model, epoch, split='Test')

#%%
xe_avg, accuracy = validate(cifar10_test_dataset, ema_model, epoch,split='test')
#%%
xe_loss, l2u_loss, total_loss, accuracy = train(trainX, trainU, model, ema_model, optimizer, epoch)
