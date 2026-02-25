# MbaGCN

This is the official implementation of the following paper:

> [Mamba-Based Graph Convolutional Networks: Tackling Over-smoothing with Selective State Space](https://arxiv.org/abs/2501.15461)
> 
> Accepted by IJCAI 2025

<div align="center">
  <img src="https://github.com/hexin5515/MbaGCN/blob/main/Image/MbaGCN.jpg" width="1600px"/>
</div>

## Environment Setup

**Required Dependencies** :

* torch>=2.1.2
* torch_geometric>=2.5.2
* python>=3.8
* scipy>=1.12.0
* numpy>=1.23.5

## Quick Start

**Actor Dataset**

The main experiments:
```
cd NodeClassification/

python training.py --dataset Actor --lr 0.001 --Stat_lr 0.01 --mamba_dropout 0.6 --weight_decay 5e-3 --d_model 64 --d_inner 64 --dt_rank 4 --d_state 4 --bias True --layer_num 11 --net GCN_mamba_Net
```
Note: The dataset will be automatically downloaded when the code is executed

## Citation
If you find our repository useful for your research, please consider citing our paper:
```
@inproceedings{ijcai2025p595,
  title     = {Mamba-Based Graph Convolutional Networks: Tackling Over-smoothing with Selective State Space},
  author    = {He, Xin and Wang, Yili and Fan, Wenqi and Shen, Xu and Juan, Xin and Miao, Rui and Wang, Xin},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {5345--5353},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/595},
  url       = {https://doi.org/10.24963/ijcai.2025/595},
}
```
