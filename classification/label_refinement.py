import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']


data_path = './'
train_lst_path = data_path + 'data/train_cls.txt'

seg_path = '/your_dir/segmentation/data/scores/voc12/deeplabv2_resnet101_msc/train_aug/trainaug_pred/'
init_label_path =  data_path + 'pseudo_labels/'


save_path = './refined_labels/'

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

# generate proxy ground-truth
def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    seg_name = seg_path + name + '.png'
    init_label_name = init_label_path + name + '.png'

    
    if not os.path.exists(seg_name):
        print('seg_name is wrong')
        return
    if not os.path.exists(init_label_name):
        print('init_label_name is wrong')
        return

    gt = np.asarray(Image.open(seg_name), dtype=np.int32)
    init_label = np.asarray(Image.open(init_label_name), dtype=np.int32)


    height, width = gt.shape

    all_cat = np.unique(gt).tolist()

    cls_label = []
    cls_label.append(0)
    for i in range(len(fields) - 1):
        k = i + 1
        category = int(fields[k]) + 1
        cls_label.append(category)
    
    cls_new = set(all_cat).difference(set(cls_label))
    cls_new = list(cls_new)

    if len(cls_new):
        for i in range(len(cls_new)):                  
            gt[gt==cls_new[i]] = 255                 #if other class exist, set 255


    if len(fields) > 2:                         #with background disagreement, select the initial label
        flag = ((gt == 0) & (init_label != 0))     
        gt = np.where(flag, init_label, gt)

    flag_missing = 0
    for i in range(len(fields) - 1):
        k = i + 1
        category = int(fields[k]) + 1
        cat_exist = (gt==category)
        if cat_exist.sum() == 0: 
            flag_missing = 1
    
    if flag_missing:          #if class is missing, bg is 255
        gt[gt==0] = 255


    # we ignore the whole image for an image with a small ratio of semantic objects
    
    out = gt 
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        out[...] = 255

    # output the proxy labels using the VOC12 label format
    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)

### Parallel Mode
pool = multiprocessing.Pool(processes=16)
pool.map(gen_gt, range(len(lines)))
#pool.map(gen_gt, range(100))
pool.close()
pool.join()

