from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess
from models import *

import matplotlib.pyplot as plt

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--outpath', default='out',
                    help='output folder')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

if not os.path.exists(args.outpath):
    os.mkdir(args.outpath)

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()

        imgL, imgR= Variable(imgL,volatile=True), Variable(imgR, volatile=True)

        # with torch.no_grad():
        pred, cost = model(imgL,imgR)
        pred = torch.squeeze(pred)
        cost = cost.squeeze()
        pred_disp = pred.data.cpu().numpy()
        pred_cost = cost.data.cpu().numpy()
        return pred_disp, pred_cost


def main():
   processed = preprocess.get_transform(augment=False)
   min_disp = 80
   sample_frame_id = 365
   for inx in range(sample_frame_id,sample_frame_id+1):
   # for inx in range(3244):
       imgLfile = os.path.join(args.datapath, 'left/%07d.png' % (inx+1))
       imgRfile = os.path.join(args.datapath, 'right/%07d.png' % (inx+1))
       imgL_o = (skimage.io.imread(imgLfile).astype('float32'))
       imgR_o = (skimage.io.imread(imgRfile).astype('float32'))
       imgL_o = skimage.transform.resize(imgL_o, (384,1248), preserve_range=True)
       imgR_o = skimage.transform.resize(imgR_o, (384,1248), preserve_range=True)
       imgL_o = imgL_o[:,:-min_disp,:]
       imgR_o = imgR_o[:,min_disp:,:]
       imgL_o = skimage.transform.resize(imgL_o, (384,1248), preserve_range=True)
       imgR_o = skimage.transform.resize(imgR_o, (384,1248), preserve_range=True)
       plt.subplot(4,1,1)
       plt.imshow(imgL_o.astype('uint8'))
       plt.subplot(4,1,2)
       plt.imshow(imgR_o.astype('uint8'))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
       # print(imgL.shape)

       # pad to (384, 1248)
       # top_pad = 384-imgL.shape[2]
       # left_pad = 1248-imgL.shape[3]
       # imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       # imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp, pred_cost = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))
       print(pred_disp.max())
       print(pred_disp.min())
       # top_pad   = 384-imgL_o.shape[0]
       # left_pad  = 1248-imgL_o.shape[1]
       # img = pred_disp[top_pad:,:-left_pad]
       img = pred_disp
       plt.subplot(4,1,3)
       plt.imshow(img)
       plt.colorbar()
       plt.subplot(4,1,4)
       plt.plot(pred_cost[:,200,600].flatten())
       plt.show()
       skimage.io.imsave(os.path.join(args.outpath,'%07d.png'%(inx)),
                        (img*256).astype('uint16'))

if __name__ == '__main__':
   main()
