## mixmatch_tensorflow
This is an unofficial Tensorflow implementation of MixMatch:[MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249). 


### Results
#### CIFAR-10
|  Implementation/Labels  |     250    |     500    |    1000    |    2000    |    4000    | 
| ----------------------- |-----------:|-----------:|-----------:|-----------:|-----------:|
| MixMatch Paper          | 88.92±0.87 | 90.35±0.94 | 92.25±0.32 | 92.97±0.15 | 93.76±0.06 |
| mixmatch-tensorflow  |       |            |            |            |            |

### Prerequisites
pip installs:
~~~
numpy>=1.17.2
pyyaml>=5.1.2
tensorflow>=2.0
tensorflow-datasets>=1.2.0
tqdm>=4.36.1
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

