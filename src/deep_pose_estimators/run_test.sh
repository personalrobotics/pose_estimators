#!/bin/bash

python test.py \
    --gpus "0" \
    --num_classes 49 \
    --img_path "datasets/food_all_images/setc_02820.jpg" \
    --checkpoint "checkpoint/food_ckpt.pth" \
    --label_map "datasets/food_label_map.pkl"

