modelpath=models/pretrained_model_KITTI2015.tar
# modelpath=models/pretrained_sceneflow.tar
# modelpath=trained/finetune_50.tar

datapath=/home/chaoyang/datasets/starwar_trailer_1080p/
python2 test.py --maxdisp 192 \
                     --model stackhourglass \
                     --datapath $datapath \
                     --loadmodel $modelpath \
                     --outpath starwar_trailer_1080p_result
