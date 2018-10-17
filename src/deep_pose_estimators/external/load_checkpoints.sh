#!/bin/bash

echo "load checkpoints for spnet"
wget https://personalrobotics.cs.washington.edu/files/food_manipulation/bite_selection_package/checkpoint/food_spnet_c6_dense_3_6_a_18_ckpt.pth -P ./bite_selection_package/checkpoint/

echo "load checkpoints for retinanet"
wget https://personalrobotics.cs.washington.edu/files/food_manipulation/pytorch_retinanet/checkpoint/food_c6_ckpt.pth -P ./pytorch_retinanet/checkpoint/
wget https://personalrobotics.cs.washington.edu/files/food_manipulation/pytorch_retinanet/pretrained/food_c6_net.pth -P ./pytorch_retinanet/pretrained/
