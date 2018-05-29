#!/bin/bash

python convert_tf_dataset.py \
    --prefix food \
    --img_base_dir ~/hard_data/Data/foods/data/images \
    --ann_base_dir ~/hard_data/Data/foods/data/annotations/xmls \
    --label_map_filename ~/hard_data/Data/foods/data/food_label_map.pbtxt

