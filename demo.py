import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import argparse
from matplotlib import pyplot as plt
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from data import VOC_CLASSES as labels

parser = argparse.ArgumentParser(description='Single Shot Detection Demo')
parser.add_argument('-i', '--img', type=str, dest='image_path',
        default='demo/street.jpg', help='Image path')
parser.add_argument('-cp', '--checkpoint', type=str, dest='checkpoint',
        default=os.path.join('checkpoints', 'ssd300_VOC_115000.pth'),  help='Checkpoint path')
parser.add_argument('-rgb', '--mean_rgb', type=str, dest='mean_rgb',
        default='104.0, 117.0, 123.0',  help='Mean value of RGB channels')
parser.add_argument('-d', '--data_shape', type=int, dest='data_shape',
        default=300,  help='Size of network input (not the actual image size)')
parser.add_argument('-n', '--num_class', type=int, dest='num_class',
        default=21,  help='Number of classes (including ground truth)')
args = parser.parse_args()


# Initialize SSD
net = build_ssd('test', args.data_shape, args.num_class)
net.load_weights(args.checkpoint)

# Load image
image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

x = cv2.resize(image, (args.data_shape, args.data_shape)).astype(np.float32)
mean_rgb = [float(n) for n in args.mean_rgb.split(',')]
x -= tuple(mean_rgb)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)

# wrap tensor in Variable
xx = Variable(x.unsqueeze(0))
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

plt.figure(figsize=(6,6))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1

plt.show()



