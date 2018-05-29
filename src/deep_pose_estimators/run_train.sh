#!/bin/bash

if [ ! -f ./pretrained/net.pth ]; then
    echo "Run 'init_retinanet.py' first by executing:"
    echo ""
    echo "    python init_retinanet.py"
    echo ""
    echo "and check if the net file is generated in pretrained/."
    exit
fi

python3 train.py \
    --gpus "2 3" \
    --num_classes 49 \
    --train_batch_size 8 \
    --test_batch_size 4 \
    --net "pretrained/food_net.pth" \
    --img_dir "datasets/food_all_images" \
    --train_list "datasets/food_ann_train.txt" \
    --test_list "datasets/food_ann_test.txt" \
    --checkpoint "checkpoint/food_ckpt.pth"

