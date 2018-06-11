# -*- coding: utf-8 -*-
import os
import math
import random

import numpy as np
import tensorflow as tf
import socket
import cv2

slim = tf.contrib.slim

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
#from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing


# Main image processing routine.
#def process_image(img, select_threshold=0.15, nms_threshold=0.25, net_shape=(512, 512)):
def process_image(img, select_threshold=0.152, nms_threshold=0.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# UDP socket 통신으로 Detecting 결과인 x, y, w, h 전송
def udp_send_data(socket, UDP, xmin, ymin, xmax, ymax):
    # sx:0000,sy:0000,ex:0000,ey:0000 (31 자리) start, end
    data = 'sx:'+str(xmin)+",sy:"+str(ymin)+",ex:"+str(xmax)+",ey"+str(ymax)
    socket.sendto(data.ljust(31), UDP) # ljust : 왼쪽 정렬 (비어 있는 곳은 공백 처리)

# UDP socket 통신으로 이미지를 전송한다.
def udp_send_img(socket, UDP, image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, img_encode = cv2.imencode('.jpg', image, encode_param)

    data = np.array(img_encode)
    data = data.tostring()
    socket.sendto(str(len(data)).ljust(16), UDP) # 이미지를 Matrix 전환시 데이터의 길이
    socket.sendto(data, UDP)


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Server Parameter
UDP_IP = '192.168.0.51'
UDP_PORT = 9999

#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
#UDP = (UDP_IP, UDP_PORT)
#s.bind(UDP) # bind


# Input placeholder.
net_shape = (300, 300)
#net_shape = (512, 512)
data_format = 'NHWC'#'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
#ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
#ckpt_filename = 'checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Test on some demo image and visualize output.
#path = "rtsp://61.108.209.254/L01154/video1"
path = 'rtsp://61.108.209.254/L01154/video1'
path = 'rtsp://61.108.209.254/L01150/video1'
#path = 'rtsp://61.108.209.254/L01288/video1'
#path = 'rtsp://61.108.209.254/L01077/video1'
#path = 'rtsp://61.108.209.254/L01207/video1'
#path = 'rtsp://61.108.209.254/L01208/video1'
#image_names = sorted(os.listdir(path))

capture = cv2.VideoCapture(path)

while(capture.isOpened()):
    ret, frame = capture.read()
    width = capture.get(3)
    height = capture.get(4)

    if ret < 0:
        break
    frame = cv2.resize(frame,(720, 480))
    #udp_send_img(s, UDP, frame) # UDP 이미지 전송
    rclasses, rscores, rbboxes = process_image(frame)

    height = frame.shape[0] # 사진의 높이
    width = frame.shape[1] # 사진의 너비
    print "#----------------------------------------"
    print(rclasses)
    idx = 0
    for d_class in range(len(rclasses)):
        # result = class
        result = int(rclasses[d_class])
        # 6 : Bus, 7 : Car
        if result == 7 or result == 6:
            i = d_class
            ymin = int(rbboxes[i][0] * height)
            xmin = int(rbboxes[i][1] * width)
            ymax = int(rbboxes[i][2] * height)
            xmax = int(rbboxes[i][3] * width)
            cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),(0,255,0),2)
            cv2.rectangle(frame,(xmin,ymin+5),(xmax, ymin),(1,1,1),-1)
            cv2.putText(frame, "car" + str(idx) ,(xmin+15,ymin+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) ,2)
            print "idx ", str(idx), " object : car ", xmin, ", ", xmax, ", ", ymin, ", ", ymax
            #udp_send_data(socket, UDP, xmin, ymin, xmax, ymax) # UDP Detecting 결과 전송
	    idx += 1
        else:
            continue

    cv2.imshow('result',frame)
    cv2.waitKey(1)


