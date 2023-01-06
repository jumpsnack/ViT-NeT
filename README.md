# ViT-NeT: Interpretable Vision Transformers with Neural Tree Decoder

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-net-interpretable-vision-transformers/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=vit-net-interpretable-vision-transformers)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-net-interpretable-vision-transformers/image-classification-on-inaturalist)](https://paperswithcode.com/sota/image-classification-on-inaturalist?p=vit-net-interpretable-vision-transformers)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-net-interpretable-vision-transformers/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=vit-net-interpretable-vision-transformers)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-net-interpretable-vision-transformers/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=vit-net-interpretable-vision-transformers)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-net-interpretable-vision-transformers/fine-grained-image-classification-on-stanford)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford?p=vit-net-interpretable-vision-transformers)

This repository contains PyTorch code for ViT-NeT. It builds on code from the [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
![fig1](./figs/fig%20s-1.png)

# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [apex](https://github.com/NVIDIA/apex):

```
pip install pytorch pytorch torchvision
```

## Data preparation

Download and extract CUB_200_2011 images from https://drive.google.com/drive/folders/1cnQHqa8BkVx90-6-UojHnbMB0WhksSRc.

```
/path/to/data_root/
  CUB_200_2011/
    attributes/
    dataset/
      test_crop/
        class1/
          img1.jpg
      test_full/
        class1/
          img1.jpg
      train_corners/
        class1/
          img1.jpg
      train_crop
        class1/
          img1.jpg
  train_test_split.txt
```

## Evaluation
To evaluate ViT-NeT on CUB_200_2011 test set, run:
```
python main.py --eval --resume /path/to/weight --cfg configs/swin_tree_patch4_window14_448_CUB.yaml
```
ViT-NeT CUB Pretrained Model [GoogleDrive](https://drive.google.com/file/d/1n-54lU0Tr0WXbn1E2geZhmfaXLoiRXxj/view?usp=sharing)

## Training
To train ViT-NeT on CUB_200_2011 on a single node with 4 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --cfg configs/swin_tree_patch4_window14_448_CUB.yaml --batch-size 16
```
SwinT/B-224-ImageNet-22K Pretrained Model (https://github.com/microsoft/Swin-Transformer)
