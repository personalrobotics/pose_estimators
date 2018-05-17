#!/bin/bash

python convert_tf_dataset.py \
    --prefix tf \
    --img_base_dir "image directory path of tf dataset" \
    --ann_base_dir "annotation directory path of tf dataset" \
    --label_map_filename "label_map.pbtxt path of tf dataset"

