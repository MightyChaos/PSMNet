#!/bin/bash

# python main.py --maxdisp 192 \
#                --model stackhourglass \
#                --datapath /media/jiaren/ImageNet/SceneFlowData/ \
#                --epochs 0 \
#                --loadmodel ./trained/checkpoint_10.tar \
#                --savemodel ./trained/



python2 finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath /home/chaoyang/datasets/kitti_scene_flow/training/ \
                   --epochs 300 \
                   --loadmodel ./models/pretrained_sceneflow.tar \
                   --savemodel ./trained/
