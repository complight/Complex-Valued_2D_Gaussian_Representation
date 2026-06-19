#!/bin/bash

# Run the parallel training script with parser default values
python train_2d_gaussian_numerical_parallel.py \
  --target_image_path ./images/071.png \
  --depth_path ./images/d_071.png \
  --compression_ratio 0.2 \
  --num_itrs 2001 \
  --viz_freq -1 \
  --eval_freq 2000 \
  --img_size 1024 640 \
  --split_ratio 1.0 \
  --gaussian_weights_path "" \
  --device cuda
