# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:40:12 2017

@author: ljm
"""

import re
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset = 'dataset'
for filename in os.listdir(dataset) :
    img = Image.open(dataset + '/' + filename)
    img = img.resize((256, 256))
    path = dataset + '/resize/'
    if os.path.isdir(path) :
        path
    else :
        os.mkdir(path)
    #f = open(path + '.jpg', 'rb')
    #f.write(img)
    img.save(path + filename)
    #img.show()
    
print('done')