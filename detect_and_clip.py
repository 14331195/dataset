# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:14:29 2017

@author: ljm

该文件用于裁剪出图片数据中的空调部分
"""
import os, sys
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class TOD(object) :
    def __init__(self):
        self.PATH_TO_CKPT = r'D:\Anaconda\mydetector\TFrecord\output2\frozen_inference_graph.pb' #已经训练好的识别图片中空调的模型路径
        self.PATH_TO_LABELS = r'D:\Anaconda\mydetector\dataset\pet_label_map.pbtxt' #训练空调模型时的标签文件
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
                
                
                # 用于可视化检测结果.
#                vis_util.visualize_boxes_and_labels_on_image_array(
#                        image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
#                        np.squeeze(scores), self.category_index, 
#                        use_normalized_coordinates=True, line_thickness=8)
                tmp = np.squeeze(boxes)
                box = tuple(tmp[0].tolist())
                return (box, image)
                
if __name__ == '__main__':
#    img = cv2.imread(r'D:\Anaconda\mydetector\images\dataset\格力（GREE）大1匹 变频冷暖 品悦 壁挂式空调(清爽白) KFR-26GW#(26592)FNhAa-A3\0.jpg')
#    cv2.imshow('ss', img)
#    cv2.waitKey(0)
    detector = TOD()
   
    filepath = r'D:\Anaconda\mydetector\images\dataset' #待裁剪的图片数据路径
    savepath = r'D:\Anaconda\mydetector\images\dataset\clip_image' #裁剪后结果保存路径
    dir_list = os.listdir(filepath)
    dir_list.sort()
    
#    image = cv2.imread(r'D:\Anaconda\mydetector\images\7.jpg')
#    shape = image.shape
#    (box, img) = detector.detect(image)
#    ymin = int(shape[0] * box[0])
#    xmin = int(shape[1] * box[1])
#    ymax = int(shape[0] * box[2])
#    xmax = int(shape[1] * box[3])
#    print(xmin, ymin, xmax, ymax)
#    crop_img = image[ymin:ymax, xmin:xmax]
#    cv2.imwrite('images/' + 'tmp7.jpg', crop_img)
    
    i = 0
    img_index = -1
    for img_dir in dir_list:    
        if i <= 55:     #已经裁剪好的目录直接跳过
            i += 1
            img_index += 1
            continue
        if os.path.isfile(filepath + '\\' + img_dir):
            continue
        
        #每个类型的空调图片 创建一个对应的文件夹目录
        if os.path.isdir(savepath+'\\'+str(img_index)):
            pass
        else:
            os.mkdir(savepath+'\\' + str(img_index))
            f = open(filepath + '\\' + 'map.txt', 'a')  #将类型与索引建立对应关系并保存
            f.writelines(str(img_index) + '|' + img_dir + '\n')
            f.close()
            
        print(img_dir)
        img_list = os.listdir(filepath + '\\' + img_dir)
        k = 0
        for img_file in img_list:
            #先判断当前图片文件是否已经截取过，若已处理则跳过
            file = os.path.join(savepath, str(img_index), str(k) + '.jpg')
            if os.path.isfile(file):
                #print('pass')
                k += 1
                continue
            
            print(img_file)
            path = os.path.join(filepath, img_dir, img_file)
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            shape = image.shape
            
            (box, img) = detector.detect(image)
            ymin = int(shape[0] * box[0])
            xmin = int(shape[1] * box[1])
            ymax = int(shape[0] * box[2])
            xmax = int(shape[1] * box[3])
            print(xmin, ymin, xmax, ymax)
            crop_img = image[ymin:ymax, xmin:xmax]
            cv2.imwrite('images/dataset/clip_image/' + str(img_index) + '/' + str(k) + '.jpg', crop_img)
            k += 1
            
        img_index += 1
            
        
                
        
