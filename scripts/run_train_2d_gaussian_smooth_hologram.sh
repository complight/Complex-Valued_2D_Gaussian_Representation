#!/bin/bash

python train_2d_gaussian.py \
  --target_image_path ./images/tiger.jpg \
  --depth_path ./images/tiger_depth.png \
  --hologram_type phase-only \
  --compression_ratio 0.2 \
  --num_itrs 2001 \
  --viz_freq -1 \
  --eval_freq 2000 \
  --lr 0.01 \
  --img_size 1024 640 \
  --split_ratio 1.0 \
  --device cuda
