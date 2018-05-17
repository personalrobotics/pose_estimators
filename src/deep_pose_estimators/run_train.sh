#!/bin/bash

python3 train.py \
    --gpus "0" \
    --num_classes 49 \
    --train_batch_size 8 \
    --test_batch_size 4 \
    --img_dir "data/food_all_images" \
    --train_list "data/food_ann_train.txt" \
    --test_list "data/food_ann_test.txt" \
    --checkpoint "checkpoint/food_ckpt.pth"
