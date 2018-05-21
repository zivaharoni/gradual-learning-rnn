# Gradual Learning of Recurrent Neural Networks

## Intro

We present an implementation of the current state-of-the-art algorithm on Penn Treebank and WikiText-2 text prediction corpuses.

Described in the article:
[Gradual Learning of Recurrent Neural Networks](https://arxiv.org/abs/1708.08863)

By Ziv Aharoni, Gal Rattner and Haim Permutter

Preprint 2018

## Requirements

Python 3.6, PyTorch 0.2.0

## Download the data

```./get_data.sh```

## Results reproduction

### Penn Treebank

Train the model:

```python run-gl-lwgc.py```

*See run-gl-lwgc.py script for hyper-parameters and configurations.
Script log files are saved by default under running directory in './GL/Lx/TEST/', the logging path can be changed by setting 'dirs' under general serrings in the script

### Penn Treebank (multi-GPUs)

Set number of GPUs using CUDA_VISIBLE_DEVICES variable:

```CUDA_VISIBLE_DEVICES=0,1,2 ```

 then change gpu settings in run-gl-lwgc.py script to a list of gpu numbers (for example 'gpu=0,1,2') and run the script:

```python run-gl-lwgc.py```

## Acknowledgements

Our code is based on the implementation of Zhilin Yang, Zihang Dai et. al., found in the repo:
https://github.com/zihangdai/mos


