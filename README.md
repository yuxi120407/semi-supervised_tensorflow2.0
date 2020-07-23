##  Semi-Supervised Learning tensorflow2.0
This is an Tensorflow implementation of semi-supervised learning with the following methods: [Pseudo-label](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf), [Pi_model](https://arxiv.org/abs/1610.02242), [VAT](https://arxiv.org/abs/1704.03976), [mean_teacher](https://arxiv.org/abs/1703.01780), [Mixup](https://arxiv.org/abs/1710.09412), [ICT](https://www.ijcai.org/Proceedings/2019/0504.pdf) and [Mixmatch](https://arxiv.org/abs/1905.02249).


### Results
#### CIFAR-10
|  Implementation/Labels  |     250    |     500    |    1000    |    2000    |    4000    | 
| ----------------------- |-----------:|-----------:|-----------:|-----------:|-----------:|
| MixMatch Paper          | 88.92±0.87 | 90.35±0.94 | 92.25±0.32 | 92.97±0.15 | 93.76±0.06 |
| mixmatch-tensorflow     |            |            |            |            |     93.23±0.12|    

### Prerequisites
pip installs:
~~~
numpy>=1.17.2
tensorflow-gpu>=2.0
tensorflow-datasets>=1.2.0
~~~

### Hyperparameters
~~~
Epoch = 1024
lr = 0.002
batch-size = 64
~~~

### Citations
~~~
@misc{berthelot2019mixmatch,
    title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
    author={David Berthelot and Nicholas Carlini and Ian Goodfellow and Nicolas Papernot and Avital Oliver and Colin Raffel},
    year={2019},
    eprint={1905.02249},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
~~~

