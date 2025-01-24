# MbaGCN

This is the official implementation of the following paper:

Mamba-Based Graph Convolutional Networks: Tackling Over-smoothing with Selective State Space

<div align="center">
  <img src="https://github.com/hexin5515/MbaGCN/blob/main/Image/MbaGCN.jpg" width="1200px"/>
</div>

## Environment Setup

**Required Dependencies** :

* torch>=2.1.2
* torch_geometric>=2.5.2
* python>=3.8
* scikit-learn>=1.4.1
* networkx>=2.7
* rdkit>=2024.3.2
* dgl>=2.1.0

## Quick Start

**Actor Dataset**

The main experiments:
```
cd NodeClassification/

python training.py --dataset Actor --lr 0.001 --Stat_lr 0.01 --mamba_dropout 0.6 --weight_decay 5e-3 --d_model 64 --d_inner 64 --dt_rank 4 --d_state 4 --bias True --layer_num 11 --net GCN_mamba_Net_New
```
