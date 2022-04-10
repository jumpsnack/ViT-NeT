import argparse
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .node import Node
import cv2
import matplotlib.pyplot as plt
import os
from math import sqrt
from timm.models.layers import trunc_normal_


def find_high_activation_crop(mask, threshold):
    threshold = 1. - threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


class GCB(nn.Module):
    mean = 0.5
    std = 0.1

    def __init__(self, dim, patch_dim=49):
        super(GCB, self).__init__()

        exp_dim = int(dim * 1.)

        self.cm = nn.Linear(dim, 1)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.cm(patches)
        x = torch.bmm(h.permute(0, 2, 1), F.softmax(x, 1)).squeeze(-1)
        x = self.wv1(x)
        x = self.gelu(self.norm(x))
        x = self.wv2(x)
        x = h + x.unsqueeze(1)
        x = self.ffn_norm(x)
        x = F.sigmoid(x)
        return x



class Branch(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, l: Node, r: Node, proto_size:list):
        super(Branch, self).__init__(index)
        self.l = l
        self.r = r
        self.img_size = 448
        self.gcb_l = GCB(256)
        self.gcb_r = GCB(256)
        self.max_score = float('-inf')
        self.proto_size = proto_size

    def forward(self, logits, patches, **kwargs):
        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)

        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        similarity = torch.log((distance + 1) / (distance + 1e-4))

        self.maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1).squeeze(-1)
        to_left = self.maxim[:, 0]
        to_right = 1 - to_left

        l_dists, _ = self.l.forward(logits, self.gcb_l(patches), **kwargs)
        r_dists, _ = self.r.forward(logits, self.gcb_r(patches), **kwargs)

        if torch.isnan(self.maxim).any() or torch.isinf(self.maxim).any():
            raise Exception('Error: NaN/INF values!', self.maxim)

        node_attr[self, 'ps'] = self.maxim
        node_attr[self.l, 'pa'] = to_left * pa
        node_attr[self.r, 'pa'] = to_right * pa
        return to_left.unsqueeze(-1) * l_dists + to_right.unsqueeze(-1) * r_dists, node_attr

    def extract_internal_patch(self, similarities, x_np, node_id):
        distance_batch = similarities[0][0]

        distance_batch[distance_batch < distance_batch.max()] = distance_batch.min()
        sz, _ = distance_batch.shape
        distance_batch = distance_batch.view(sz, sz, 1)
        similarity_map = distance_batch.detach().cpu().numpy()

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_sim_map), cv2.COLORMAP_JET)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[..., ::-1]
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(
            similarity_map)] = 0  # mask similarity map such that only the nearest patch z* is visualized

        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                 dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, 0.98)
        high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]

        plt.imsave(
            fname=f'node_interp/node-{node_id}-patch.png',
            arr=high_act_patch, vmin=0.0, vmax=1.0)

        imsave_with_bbox(
            fname=f'node_interp/node-{node_id}.png',
            img_rgb=x_np,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3], color=(0, 0, 255))

    def explain_internal(self, logits, patches, x_np, node_id, **kwargs):
        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)

        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        similarity = torch.log((distance + 1) / (distance + 1e-4))

        self.maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1).squeeze(-1)
        to_left = self.maxim[:, 0]
        to_right = 1 - to_left

        out_map = kwargs['out_map']

        score = self.maxim.squeeze().item()
        if score >= self.max_score:
            self.max_score = score
            similarity = F.conv2d(similarity, weight=torch.ones((1, 1, 2, 2)).cuda(),
                                  stride=2) if self.proto_size == [1,1] else similarity
            self.extract_internal_patch(similarity, x_np, node_id)

        l_dists, _ = self.l.explain_internal(logits, self.gcb_l(patches), x_np, node_id * 2 + 1, **kwargs)
        r_dists, _ = self.r.explain_internal(logits, self.gcb_r(patches), x_np, node_id * 2 + 2, **kwargs)

        if torch.isnan(self.maxim).any() or torch.isinf(self.maxim).any():
            raise Exception('Error: NaN/INF values!', self.maxim)

        node_attr[self, 'ps'] = self.maxim
        node_attr[self.l, 'pa'] = to_left * pa
        node_attr[self.r, 'pa'] = to_right * pa
        return to_left.unsqueeze(-1) * l_dists + to_right.unsqueeze(-1) * r_dists, node_attr

    def _l2_conv(self, x, proto_vector, stride, dilation, padding):
        B, L, C = x.shape
        W = int(sqrt(L))
        x = x.permute(0, 2, 1).view(B, C, W, W)
        _, _, k, _ = proto_vector.shape
        ones = torch.ones_like(proto_vector, device=x.device)

        x2 = x ** 2
        x2_patch_sum = F.conv2d(x2, weight=ones, stride=stride, dilation=dilation, padding=padding)

        p2 = proto_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(x, weight=proto_vector, stride=stride, dilation=dilation, padding=padding)
        intermediate_result = -2 * xp + p2_reshape

        distances = torch.sqrt(F.relu(x2_patch_sum + intermediate_result))

        return distances

    def g(self, **kwargs):
        out_map = kwargs['out_map']  # Obtain the mapping from decision nodes to conv net outputs
        conv_net_output = kwargs['conv_net_output']  # Obtain the conv net outputs
        out = conv_net_output[out_map[self.index]]  # Obtain the output corresponding to this decision node
        return out.squeeze(dim=1)

    def plot_map(self, node_id, similarities, prefix, r_node_id, x_np, pool_map, direction, correctness):
        distance_batch = similarities[0][0]

        distance_batch[distance_batch < distance_batch.max()] = distance_batch.min()
        sz, _ = distance_batch.shape
        distance_batch = distance_batch.view(sz, sz, 1)
        similarity_map = distance_batch.detach().cpu().numpy()

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_sim_map), cv2.COLORMAP_JET)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[..., ::-1]
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(
            similarity_map)] = 0  # mask similarity map such that only the nearest patch z* is visualized

        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                 dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        plt.imsave(
            fname=os.path.join('heatmap', prefix, '%s_masked_upsampled_heatmap_%s.png' % (str(r_node_id), direction)),
            arr=upsampled_prototype_pattern, vmin=0.0, vmax=1.0)

        similarity_heatmap = cv2.resize(similarity_heatmap,
                                        dsize=(self.img_size, self.img_size))
        plt.imsave(fname=os.path.join('heatmap', prefix,
                                      '%s_heatmap_latent_similaritymap_%s.png' % (str(r_node_id), direction)),
                   arr=similarity_heatmap, vmin=0.0, vmax=1.0)

        upsampled_act_pattern = cv2.resize(masked_similarity_map,
                                           dsize=(self.img_size, self.img_size),
                                           interpolation=cv2.INTER_CUBIC)
        rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
        rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        overlayed_original_img = 0.5 * x_np + 0.4 * heatmap
        plt.imsave(
            fname=os.path.join('heatmap', prefix, '%s_heatmap_original_image_%s.png' % (str(r_node_id), direction)),
            arr=overlayed_original_img, vmin=0.0, vmax=1.0)

        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, 0.98)
        high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]
        plt.imsave(
            fname=os.path.join('heatmap', prefix, '%s_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
            arr=high_act_patch, vmin=0.0, vmax=1.0)

        imsave_with_bbox(
            fname=os.path.join('heatmap', prefix,
                               '%s_bounding_box_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
            img_rgb=x_np,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255) if not correctness else (0, 0, 255))

    def hard_forward(self, logits, patches, **kwargs):
        batch_size = patches.size(0)
        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)
        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        similarity = torch.log((distance + 1) / (distance + 1e-4))

        self.maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1).squeeze(-1)
        to_left = self.maxim[:, 0]
        to_right = 1 - to_left

        if to_left > to_right:
            l_dists, _ = self.l.hard_forward(logits, self.gcb_l(patches), **kwargs)
            return l_dists, node_attr
        else:
            r_dists, _ = self.r.hard_forward(logits, self.gcb_r(patches), **kwargs)

            return r_dists, node_attr

    def explain(self, logits, patches, l_distances, r_distances, x_np, y, prefix, r_node_id, pool_map,
                **kwargs):
        ps = self.g(**kwargs)

        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        similarity = torch.log((distance + 1) / (distance + 1e-4))

        maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1)
        similarity = F.conv2d(similarity, weight=torch.ones((1, 1, 2, 2)).cuda(),
                              stride=2) if self.proto_size == [1, 1] else similarity
        out_map = kwargs['out_map']
        node_id = out_map[self.index]
        self.patch_size = kwargs['img_size']

        to_left = maxim[:, 0]
        to_right = 1 - to_left

        print(f'{node_id} -> ', flush=True)
        print(to_left)
        if to_left > to_right:
            self.plot_map(node_id, similarity, prefix, r_node_id, x_np, pool_map, 'left', True)
            return self.l.explain(logits, self.gcb_l(patches), l_distances, r_distances, x_np, y, prefix,
                                  r_node_id * 2 + 1, pool_map,
                                  **kwargs)

        else:
            self.plot_map(node_id, similarity, prefix, r_node_id, x_np, pool_map, 'right', True)
            return self.r.explain(logits, self.gcb_r(patches), l_distances, r_distances, x_np, y, prefix,
                                  r_node_id * 2 + 2, pool_map,
                                  **kwargs)

    @property
    def size(self) -> int:
        return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
        return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
        return {self}.union(self.l.branches).union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self, **self.l.nodes_by_index, **self.r.nodes_by_index}

    @property
    def num_leaves(self) -> int:
        return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) -> int:
        return self.l.depth + 1


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)


def im_with_bbox(img_rgb, bbox_height_start, bbox_height_end,
                 bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    return img_rgb_float
