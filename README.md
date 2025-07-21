# Multi-level Collaborative Distillation Meets Global Workspace Model: A Unified Framework for OCIL
Official implementation of the paper Multi-level Collaborative Distillation Meets Global Workspace Model: A Unified Framework for OCIL.

## 1. Requirements

The experiments are conducted using the following hardware and software:

- Hardware: NVIDIA GeForce RTX 3090 GPUs
- Software: Please refer to `requirements.txt` for the detailed package versions. Conda is highly recommended.

## 2. Datasets

### CIFAR-100
The CIFAR-100 dataset will be automatically download during the first run and stored in `./dataset/cifar100`.

### TinyImageNet
The codebase should be able to handle TinyImageNet dataset automatically and save it in the `dataset` folder. If the automatic download fails, please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip , and unzip it into `./dataset/tiny-imagenet-200`.

### ImageNet-100
Download the ImageNet dataset from [this link](http://www.image-net.org/) and follow [this](https://github.com/danielchyeh/ImageNet-100-Pytorch) for ImageNet-100 dataset generation. Put the dataset in the `./dataset/imagenet100_data` folder.

## 3. Training
### Training with a configuration file for CCL-DC framework
Training can be done by specifying the dataset path and parameters in a configuration file. The detailed commands for different datasets are as follows:

```
cifar100：
# baseline
python ./GWM-MCD/main.py --data-root-dir ./dataset/cifar100 --config ./config/BASELINE/cifar100/ER,c100,m100mbs10sbs10.yaml 
# CCL-DC
python ./GWM-MCD/main.py --data-root-dir ./dataset/cifar100 --config ./config/CCLDC/cifar100/ERCCLDC,c100,m100mbs10sbs10.yaml 
#Ours
python ./GWM-MCD/main.py --data-root-dir ./dataset/cifar100 --config ./config/OMKD/cifar100/EROMKDCCL,c100,m100mbs10sbs10.yaml 

tiny：
python ./GWM-MCD/main.py --data-root-dir ./dataset/tiny-imagenet-200 --config ./config/BASELINE/tiny/ER,tiny,m200mbs10sbs10.yaml
python ./GWM-MCD/main.py --data-root-dir ./dataset/tiny-imagenet-200 --config ./config/CCLDC/tiny/ERCCLDC,tiny,m200mbs10sbs10.yaml
python ./GWM-MCD/main.py --data-root-dir ./dataset/tiny-imagenet-200 --config ./config/OMKD/tiny/EROMKDCCL,tiny,m200mbs10sbs10.yaml

im100
python ./GWM-MCD/main.py --data-root-dir ./dataset/imagenet100_data --config ./config/BASELINE/in100/ER,in100,m200mbs10sbs10.yaml
python ./GWM-MCD/main.py --data-root-dir ./dataset/imagenet100_data --config ./config/CCLDC/in100/ERCCLDC,in100,m200mbs10sbs10.yaml
python ./GWM-MCD/main.py --data-root-dir ./dataset/imagenet100_data --config ./config/OMKD/in100/EROMKDCCL,in100,m200mbs10sbs10.yaml
```

### Training with a commend line for MOSE-MOE framework

```
cifar100：
python ./MOSE-MOE+Ours/main.py --method mose --n_tasks 10 --dataset cifar100 --buffer_size 100 --augmentation ocm
python ./MOSE-MOE+Ours/main.py --method moseccldc --n_tasks 10 --dataset cifar100 --buffer_size 100 --augmentation ocm
python ./MOSE-MOE+Ours/main.py --method moseomkdccl --n_tasks 10 --dataset cifar100 --buffer_size 100 --augmentation ocm

tiny：
python ./MOSE-MOE+Ours/main.py --method mose --n_tasks 100 --dataset tiny_imagenet --buffer_size 200 --augmentation ocm
python ./MOSE-MOE+Ours/main.py --method moseccldc --n_tasks 100 --dataset tiny_imagenet --buffer_size 200 --augmentation ocm
python ./MOSE-MOE+Ours/main.py --method moseomkdccl --n_tasks 100 --dataset tiny_imagenet --buffer_size 200 --augmentation ocm

im100
python ./MOSE-MOE+Ours/main.py --method mose --n_tasks 10 --dataset imagenet100 --buffer_size 200 --augmentation ocm
python ./MOSE-MOE+Ours/main.py --method moseccldc --n_tasks 10 --dataset imagenet100 --buffer_size 200 --augmentation ocm
python ./MOSE-MOE+Ours/main.py --method moseomkdccl --n_tasks 10 --dataset imagenet100 --buffer_size 200 --augmentation ocm
```

## TODO

- [ ] In the future, we will integrate CCL-DC and MOSE-MOE into a unified code.

## Acknowledgement

This implementation is based on the CCL-DC and MOSE-MOE framework. Special thanks to [maorong-wang](https://github.com/maorong-wang) and [Hongwei Yan](https://github.com/AnAppleCore/MOSE) for their contribution to the framework and the implementation of recent state-of-the-art methods. 
