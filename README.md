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
