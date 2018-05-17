#!/bin/bash

python3 test.py \
    --gpus "0" \
    --num_classes 49 \
    --img_path "data/food_all_images/setc_02820.jpg" \
    --checkpoint "checkpoint/food_ckpt.pth"
