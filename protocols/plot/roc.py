# =========================================================
# @ Plot File: IITD under IITD-like protocol
# =========================================================

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

from plotroc_basic import *


src_npy = ['C:\\Users\\ZhenyuZHOU\\Desktop\\Finger-Knuckle-Recognition\\output\\ConvNetVSRFNet\\ConvNet-3000-protocol.npy',
           'C:\\Users\\ZhenyuZHOU\\Desktop\\Finger-Knuckle-Recognition\\output\\ConvNetVSRFNet\\ConvNetWithoutRES-1-protocol.npy',
           'C:\\Users\\ZhenyuZHOU\\Desktop\\Finger-Knuckle-Recognition\\output\\ConvNetVSRFNet\\ConvNetWithoutRES-2-protocol.npy',
           'C:\\Users\\ZhenyuZHOU\\Desktop\\Finger-Knuckle-Recognition\\output\\ConvNetVSRFNet\\RFN-WS-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/imageblockrotationandshifted/fkv3-rfn/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/imageblockrotationandshifted/fkv3-rfn/d3-a5-topk12-protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/wholeimagerotationandshifted/fkv3-efficientnet/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/wholeshifted/fkv3-rfn/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/RFN/HD/RFN-TOP14/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/RFN/HD/RFN-TOP16/protocol3.npy',
           '/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/knuckle-recog-dcn/code/output/RFN/HD/RFN/protocol3.npy']

label = ['ConvNet-WS',
         'ConvNetWithoutRES-1',
         'ConvNetWithoutRES-2',
         'RFN-WS',
         'RFN-IBRS',
         'RFN-128-WRS-1-105-0.01',
         'EfficientNet-WRS',
         'RFN-WS',
         'RFN-128-14',
         'RFN-128-16',
         'RFN']

color = ['#000000',
         '#000080',
         '#008000',
         '#008080',
         "#c0c0c0",
         '#00ffff',
         '#800000',
         '#800080',
         '#808000',
         '#ff00ff',
         '#ff0000']
dst = 'C:\\Users\\ZhenyuZHOU\\Desktop\\Finger-Knuckle-Recognition\\output\\ConvNetVSRFNet\\ConvNet-VS-protocol3_roc.pdf'

for i in range(4):
    data = np.load(src_npy[i], allow_pickle=True)[()]
    g_scores = np.array(data['g_scores'])
    i_scores = np.array(data['i_scores'])

    print ('[*] Source file: {}'.format(src_npy[i]))
    print ('[*] Target output file: {}'.format(dst))
    print ("[*] #Genuine: {}\n[*] #Imposter: {}".format(len(g_scores), len(i_scores)))

    x, y = calc_coordinates(g_scores, i_scores)
    print ("[*] EER: {}".format(calc_eer(x, y)))

    # xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()

    lines = plt.plot(x, y, label='ROC')
    plt.setp(lines, 'color', color[i], 'linewidth', 1, 'label', label[i])

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.grid(True)
    plt.xlabel(r'False Accept Rate', fontsize=18)
    plt.ylabel(r'Genuine Accept Rate', fontsize=18)
    legend = plt.legend(loc='lower right', shadow=False, prop={'size': 16})
    plt.xlim(xmin=min(x))
    plt.xlim(xmax=0)
    plt.ylim(ymax=1)
    plt.ylim(ymin=0)

    ax=plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    plt.xticks(np.array([-4 , -2, 0]), ['$10^{-4}$', '$10^{-2}$', '$10^{0}$'], fontsize=16)
    plt.yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), fontsize=16)

if dst:
    plt.savefig(dst, bbox_inches='tight')