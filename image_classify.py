# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:14:29 2017

@author: ljm

此文件用于图像分类，对输入图像进行分类，并给出分类结果
"""
import os, sys
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class TOD(object) :
    def __init__(self):
        self.PATH_TO_CKPT = r'D:\Anaconda\mydetector\TFrecord\output2\frozen_inference_graph.pb'
        self.PATH_TO_LABELS = r'D:\Anaconda\mydetector\dataset\pet_label_map.pbtxt'
        self.NUM_CLASSES = 2
        self.detection_graph = self.load_model()
        self.category_index = self.load_label_map()
        
    def load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        return detection_graph
    
    def load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, 
                                        max_num_classes = self.NUM_CLASSES,
                                        use_display_name = True)
        category_index = label_map_util.create_category_index(categories)
        return category_index
    
    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_extended = np.expand_dims(image, axis=0)
#                print(image_np_extended)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
 
                #Actual detection
                (boxes, scores, classes, num_detections)= sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor:image_np_extended})

                tmp = np.squeeze(boxes)
                box = tuple(tmp[0].tolist())
                return (box, image)
                
if __name__ == '__main__':
#    img = cv2.imread(r'D:\Anaconda\mydetector\images\dataset\格力（GREE）大1匹 变频冷暖 品悦 壁挂式空调(清爽白) KFR-26GW#(26592)FNhAa-A3\0.jpg')
#    cv2.imshow('ss', img)
#    cv2.waitKey(0)
    detector = TOD()
   
    filepath = r'D:\Anaconda\mydetector\images\dataset'
    savepath = r'D:\Anaconda\mydetector\images\dataset\crop_image'
    dir_list = os.listdir(filepath)
    dir_list.sort()
    
    image = cv2.imread(r'D:\Anaconda\mydetector\images\test\8.jpg')
    shape = image.shape 
    (box, img) = detector.detect(image)
    ymin = int(shape[0] * box[0])
    xmin = int(shape[1] * box[1])
    ymax = int(shape[0] * box[2])
    xmax = int(shape[1] * box[3])
    print(xmin, ymin, xmax, ymax)
    crop_img = image[ymin:ymax, xmin:xmax]
    cv2.imwrite('images/test/' + 'tmp.jpg', crop_img)


    #开始识别图像
    image_path = r'images/test/tmp.jpg'

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("retrained_labels.txt")]
    
    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))