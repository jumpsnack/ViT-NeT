import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .node import Node


class Leaf(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, num_classes):
        super(Leaf, self).__init__(index)
        self.pred = nn.Linear(1024 // 4, num_classes)
        self.norm_o = nn.LayerNorm(num_classes)

    def forward(self, logits, patches, **kwargs):
        batch_size = patches.size(0)
        node_attr = kwargs.setdefault('attr', dict())

        node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        x = F.adaptive_max_pool1d(patches.permute(0, 2, 1), (1)).squeeze(-1)
        dists = self.norm_o(self.pred(x) + logits)

        self.dists = F.softmax(dists)

        node_attr[self, 'ds'] = self.dists

        return self.dists, node_attr

    def hard_forward(self, logits, patches, **kwargs):
        return self(logits, patches, **kwargs)

    def explain_internal(self, logits, patches, x_np, node_id, **kwargs):
        return self(logits, patches, **kwargs)

    def explain(self, logits, patches, l_distances, r_distances, x_np, y, prefix, r_node_id, pool_map,
                **kwargs):
        return self(logits, patches, **kwargs)

    @property
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def branches(self) -> set:
        return set()

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_branches(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0
