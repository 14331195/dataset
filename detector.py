# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:14:29 2017

@author: ljm

此文件用于检测 提取关键区域模型 的检测效果
"""

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
                
#                num_detections = '1'
                #Actual detection
                (boxes, scores, classes, num_detections)= sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor:image_np_extended})
                
                tmp = np.squeeze(boxes)
                box = tuple(tmp[0].tolist())
                print(box)
                #print(np.squeeze(scores))
                print(num_detections)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores), self.category_index, 
                        use_normalized_coordinates=True, line_thickness=8)
                
                return (box, image)
                
if __name__ == '__main__':
    detector = TOD()
    for i in range(1, 9):
        img_str = 'images/' + str(i) + '.jpg'
        image = cv2.imread(img_str)
        shape = image.shape
        print(shape)
        (box, img) = detector.detect(image)
        ymin = int(shape[0] * box[0])
        xmin = int(shape[1] * box[1])
        ymax = int(shape[0] * box[2])
        xmax = int(shape[1] * box[3])
        print(xmin, ymin, xmax, ymax)
        crop_img = image[ymin:ymax, xmin:xmax]
        cv2.imwrite('images/tmp' + str(i) + '.jpg', crop_img)
        
#        cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
#        cv2.imshow('detection', img)
#        cv2.waitKey(0)
                
        
