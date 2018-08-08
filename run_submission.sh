modelpath=models/pretrained_model_KITTI2015.tar
datapath=/home/chaoyang/datasets/kitti_scene_flow/testing/
python2 submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath $datapath \
                     --loadmodel $modelpath \
