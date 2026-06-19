#!/bin/bash

# Run the numerical hologram training script with parser default values
python train_2d_numerical.py \
  --target_image_path ./images/071.png \
  --depth_path ./images/d_071.png \
  --num_itrs 2001 \
  --viz_freq -1 \
  --eval_freq 2000 \
  --lr 0.025 \
  --img_size 1024 640 \
  --split_ratio 1.0 \
  --device cuda
