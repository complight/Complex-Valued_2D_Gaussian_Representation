#!/bin/bash

# Run the SIREN/MLP hologram training with parser defaults
python train_2d_SIREN.py \
  --base_dir ./result_2d \
  --amp_name amp_blue_cat.png \
  --phase_name phase_blue_cat.png \
  --img_size 1024 640 \
  --num_itrs 2000 \
  --lr 0.001 \
  --viz_freq 200 \
  --hidden_features 256 \
  --hidden_layers 4 \
  --model_type siren \
  --pos_enc_L 30
