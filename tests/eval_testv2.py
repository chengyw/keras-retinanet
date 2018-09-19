# -*- coding: utf-8 -*-
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import keras
import sys
import xml.etree.ElementTree as ET
import operator
import shutil
import time

sys.path.append("/home/lichunxue/Desktop/keras-retinanet-new/")
print(sys.path)
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
#from ai_site.src.api.utils import get_new_model

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from matplotlib import pyplot as plt


labels_to_names = {0:'balishui',
1:'keleguan',
2:'lvliangshengzhazhi',
3:'quecaonatie',
4:'hongniu',
5:'weitaningmengcha',
6:'nongfuNFC',
7:'jiaduobaoguan',
8:'quchenshiyuanwei',
9:'beibingyangjuqi',
10:'maidongqingningkouwei',
11:'xingbakemokawei',
12:'meirishiliulvcha',
13:'xiangcaosuda',
14:'weikabingdimoka',
15:'kalabaoweishengsu',
16:'sanyuanchunnai',
17:'lingdukele',
18:'kangshifubinghongcha',
19:'kuerlexiangli',
20:'yantaifushipingguo',
21:'feizixiaolizhi',
22:'guochanlanmei',
23:'ranchachunxiang',
24:'tianfangyetanbingfen',
25:'huoxingrusuanyuanwei',
26:'huorunmangguosuannai',
27:'qiandongyishenggulanmei',
28:'moqitaozhiyinliao',
29:'changqingyanmaihuangtao',
30:'younuolanmeiguoli',
31:'xiangpiaopiaoniurucha',
32:'weikezixiangjiao',
33:'yuanweidounaiweita'}

names_to_number1 = {
    'balishui': 0,
    'keleguan': 0,
    'lvliangshengzhazhi': 0,
    'quecaonatie': 0,
    'hongniu': 0,
    'weitaningmengcha': 0,
    'nongfuNFC': 0,
    'jiaduobaoguan': 0,
    'quchenshiyuanwei': 0,
    'beibingyangjuqi': 0,
    'maidongqingningkouwei': 0,
    'xingbakemokawei': 0,
    'meirishiliulvcha': 0,
    'xiangcaosuda': 0,
    'weikabingdimoka': 0,
    'kalabaoweishengsu': 0,
    'sanyuanchunnai': 0,
    'lingdukele': 0,
    'kangshifubinghongcha': 0,
    'kuerlexiangli':0,
    'yantaifushipingguo':0,
    'feizixiaolizhi':0,
    'guochanlanmei':0,
    'ranchachunxiang':0,
    'tianfangyetanbingfen':0,
    'huoxingrusuanyuanwei':0,
    'huorunmangguosuannai':0,
    'qiandongyishenggulanmei':0,
    'moqitaozhiyinliao':0,
    'changqingyanmaihuangtao':0,
    'younuolanmeiguoli':0,
    'xiangpiaopiaoniurucha':0,
    'weikezixiangjiao':0,
    'yuanweidounaiweita':0
}

names_to_number2 = {
    'balishui': 0,
    'keleguan': 0,
    'lvliangshengzhazhi': 0,
    'quecaonatie': 0,
    'hongniu': 0,
    'weitaningmengcha': 0,
    'nongfuNFC': 0,
    'jiaduobaoguan': 0,
    'quchenshiyuanwei': 0,
    'beibingyangjuqi': 0,
    'maidongqingningkouwei': 0,
    'xingbakemokawei': 0,
    'meirishiliulvcha': 0,
    'xiangcaosuda': 0,
    'weikabingdimoka': 0,
    'kalabaoweishengsu': 0,
    'sanyuanchunnai': 0,
    'lingdukele': 0,
    'kangshifubinghongcha': 0,
    'kuerlexiangli':0,
    'yantaifushipingguo':0,
    'feizixiaolizhi':0,
    'guochanlanmei':0,
    'ranchachunxiang':0,
    'tianfangyetanbingfen':0,
    'huoxingrusuanyuanwei':0,
    'huorunmangguosuannai':0,
    'qiandongyishenggulanmei':0,
    'moqitaozhiyinliao':0,
    'changqingyanmaihuangtao':0,
    'younuolanmeiguoli':0,
    'xiangpiaopiaoniurucha':0,
    'weikezixiangjiao':0,
    'yuanweidounaiweita':0
}

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
    
    
keras.backend.tensorflow_backend.set_session(get_session())
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#model_path = os.path.join('/home/lichunxue/Desktop/keras-retinanet/snapshots/0831/inference', 'resnet101_pascal_33.h5')
model_path = os.path.join('/home/lichunxue/Desktop/keras-retinanet-new/snapshots/0910/inference', 'resnet101_pascal_30.h5')
model = models.load_model(model_path, backbone_name='resnet101')


def init_dict():
    for k in names_to_number1.keys():
        names_to_number1[k]=0
    for k in names_to_number2.keys():
        names_to_number2[k]=0
        
def list_dir(path, list_name, extension):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                list_name.append(file_path)
    return  list_name
        
def format_result(rclasses, rscores):
    dictionary = {}
    for idx, i in enumerate(rclasses):
        if dictionary.get(i):
            dictionary[i][0] = dictionary.get(i)[0] + 1
            dictionary[i][1].append(str(rscores[idx]))
        else:
            dictionary[i] = [1, []]
            dictionary[i][1].append(str(rscores[idx]))


    res = []
    for key, item in dictionary.items():
        res.append({"goods_name": labels_to_names.get(key), "goods_cnt": item[0], "goods_conf": item[1]})
    return res

def drop_element(data, drop_idx):
    res = list(data)
    result = []
    prev = 0
    if drop_idx == []:
        return res

    for i in drop_idx:
        result += res[prev:i]
        prev = i + 1
    result += res[prev:]
    return result

def cal_area(line):
    res = (line[2] - line[0]) * (line[3] - line[1])
    return res

def drop_small_area(rbboxes):
    data = rbboxes
    area_threshold = 0.0055
    res = []
    drop_small_area_idx = []
    for idx, line in enumerate(data):
        if cal_area(line) >= area_threshold:
            res.append(data[idx])
        else:
            drop_small_area_idx.append(idx)
    return np.array(res).astype("float32"), drop_small_area_idx

def remove_mirror(rbboxes):
    data = rbboxes
    drop_list = []
    for idx, line in enumerate(data):
        if ((line[0] + line[2]) / 2) <= 0.15:
            drop_list.append(idx)

    result = []
    prev = 0
    for i in drop_list:
        result += data[prev:i]
        prev = i + 1
    result += data[prev:]
    return np.array(result).astype("float32"), drop_list

# def remove_inner_rbboxes(rbboxes):
#     overlap = {}
#     data = rbboxes
#     for idx1, l1 in enumerate(data):
#         overlap[idx1] = []
#         tl1 = (l1[1], l1[0])
#         bl1 = (l1[1], l1[2])
#         tr1 = (l1[3], l1[0])
#         br1 = (l1[3], l1[2])
#
#         for idx2, l2 in enumerate(data):
#             if not idx1 == idx2:
#                 if l2[3] < l1[1] or l2[2] < l1[0] or l2[1] > l1[3] or l2[0] > l1[2]:
#                     continue
#                 else:
#                     area1 = cal_area(l1)
#                     iou_area = cal_iou(l1, l2)
#                     iou = iou_area / float(area1)
#                     if iou >= 0.45:
#                         overlap[idx1].append(iou)
#
#     drop_list = []
#     for idx, val in overlap.items():
#         if len(val) >= 2:
#             drop_list.append(idx)
#
#     result = []
#     prev = 0
#     for i in drop_list:
#         result += data[prev:i]
#         prev = i + 1
#     result += data[prev:]
#     return result, drop_list

# remove double overlap
# def remove_overlap(rbboxes, rscores):
#     overlap = {}
#     data = rbboxes
#     for idx1, l1 in enumerate(data):
#         if idx1 not in overlap.keys():
#             overlap[idx1] = []
#         # tl1 = (l1[1], l1[0])
#         # bl1 = (l1[1], l1[2])
#         # tr1 = (l1[3], l1[0])
#         # br1 = (l1[3], l1[2])
#
#         for idx2, l2 in enumerate(data):
#             if not idx1 == idx2:
#                 if l2[3] < l1[1] or l2[2] < l1[0] or l2[1] > l1[3] or l2[0] > l1[2]:
#                     continue
#                 else:
#                     area1 = cal_area(l1)
#                     iou_area = cal_iou(l1, l2)
#                     iou = iou_area / float(area1)
#                     if iou >= 0.50:
#                         if float(rscores[idx1]) >= float(rscores[idx2]):
#                             if idx2 not in overlap.keys():
#                                 overlap[idx2] = []
#                             overlap[idx2].append(iou)
#                         else:
#                             overlap[idx1].append(iou)
#
#     drop_list = []
#     for idx, val in overlap.items():
#         if len(val) >= 2:
#             drop_list.append(idx)
#
#     result = []
#     prev = 0
#     for i in drop_list:
#         result += data[prev:i]
#         prev = i + 1
#     result += data[prev:]
#     return result, drop_list

# def cal_iou(l1, l2):
#     # l1 and l2 are [ymin, xmin, ymax, xmax]
#     # for iou x min, y min
#     ymin = l1[0] if l1[0] > l2[0] else l2[0]
#     xmin = l1[1] if l1[1] > l2[1] else l2[1]
#     ymax = l1[2] if l1[2] < l2[2] else l2[2]
#     xmax = l1[3] if l1[3] < l2[3] else l2[3]
#
#     iou_area = (ymax - ymin) * (xmax - xmin)
#     return iou_area


def cal_iou(l1,l2):
    area1 = cal_area(l1)
    area2 = cal_area(l2)
    iou_area = cal_overlap_area(l1, l2)
    iou = iou_area / (area1 + area2 - iou_area)
    return iou


# remove double overlap
def remove_double(rbboxes, rscores, rlabel):
    overlap = [[] for i in range(len(rscores))]
    data = rbboxes
    for idx1, l1 in enumerate(data):
        for idx2, l2 in enumerate(data):
            if idx2 <= idx1:
                continue
            else:
                if l2[3] < l1[1] or l2[2] < l1[0] or l2[1] > l1[3] or l2[0] > l1[2]:
                    continue
                else:
                    iou = cal_iou(l1,l2)
                    if iou >= 0.45:
                        if float(rscores[idx1]) >= float(rscores[idx2]):
                            print ("s1",rscores[idx1],"s2", rscores[idx2])
                            overlap[idx2].append(idx1)
                        else:
                            overlap[idx1].append(idx2)

    drop_list = []
    for idx in range(len(rscores)):
        if len(overlap[idx]) >= 1:
            drop_list.append(idx)

    result = []
    prev = 0
    for i in drop_list:
        result += data[prev:i]
        prev = i + 1
    result += data[prev:]
    return result, drop_list

# remove triple overlap boxes
def remove_triple(rbboxes, rscores, rlabel):
    data = rbboxes
    drop_list = []
    idx = 0
    for idx in range(len(rscores)):
        idx1 = -idx - 1
        l1 = data[idx1]
        area1 = cal_area(l1)
        overlap = []
        for idx2, l2 in enumerate(data):
            if idx2 > len(rscores)+idx1-1:
                continue
            else:
                if l2[3] < l1[1] or l2[2] < l1[0] or l2[1] > l1[3] or l2[0] > l1[2]:
                    continue
                elif idx2 not in drop_list:
                        iou_area = cal_overlap_area(l1, l2)
                        overlap.append(iou_area)
        ratio = sum(overlap)/area1
        if ratio > 0.9:
            drop_list.append(len(rscores)+idx1)
#----------------------------0612修改----------------------------------
    drop_list = reversed(drop_list)
    boxes = []
    scores = []
    labels = []
    prev = 0
    for i in drop_list:
        boxes += data[prev:i]
        scores += rscores[prev:i]
        labels += rlabel[prev:i]
        prev = i + 1
    boxes += data[prev:]
    scores += rscores[prev:]
    labels += rlabel[prev:]
    return boxes, scores, labels
#-----------------------------------------------------------------------

def cal_overlap_area(l1, l2):
    # l1 and l2 are [ymin, xmin, ymax, xmax]
    # for iou x min, y min
    ymin = l1[0] if l1[0] > l2[0] else l2[0]
    xmin = l1[1] if l1[1] > l2[1] else l2[1]
    ymax = l1[2] if l1[2] < l2[2] else l2[2]
    xmax = l1[3] if l1[3] < l2[3] else l2[3]

    iou_area = (ymax - ymin) * (xmax - xmin)
    return iou_area


def after_processing(label, score, box):

    # # drop small area
    # rbboxes, drop_small_area_idx = drop_small_area(box)
    #
    # rclasses = drop_element(label, drop_small_area_idx)
    # rscores = drop_element(score, drop_small_area_idx)
    # rclasses = np.array(rclasses)
    # rscores = np.array(rscores)
    #
    # # remove image shows up on mirror
    # rbboxes, drop_idx = remove_mirror(rbboxes)
    #
    # rclasses = drop_element(rclasses, drop_idx)
    # rscores = drop_element(rscores, drop_idx)
    # rclasses = np.array(rclasses)
    # rscores = np.array(rscores)

    # remove inner overlap

    #-----------------上一版本-------------------------------
    # box, inner_box_drop_list = remove_inner_rbboxes(box)
    #
    # label = drop_element(label, inner_box_drop_list)
    # score = drop_element(score, inner_box_drop_list)
    #
    # # remove double overlap
    # box, overlap_drop_list = remove_overlap(box, score)
    #
    # label = drop_element(label, overlap_drop_list)
    # score = drop_element(score, overlap_drop_list)
    #
    # return label, score, box
    #-------------------------------

    box, double_drop_list = remove_double(box, score, label)
    label = drop_element(label, double_drop_list)
    score = drop_element(score, double_drop_list)
    
#---------------------------0612修改----------------------------------------
    # remove triple overlap
    box, score, label = remove_triple(box, score, label)
    #label = drop_element(label, triple_drop_list)
    #score = drop_element(score, triple_drop_list)
    #print('remove_triple',box)
    #print('triple_drop_list',triple_drop_list)
#----------------------------------------------------------------------------

    return label, score, box


def res(img_path):
    # load image
    image = read_image_bgr(img_path)

    # preprocess image for network
    image = preprocess_image(image)
    image,scale = resize_image(image, 960, 1280)
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # visualize detections
    res_score = []
    res_label = []
    res_box = []
    for box, score, label in zip(boxes[0].tolist(), scores[0].tolist(), labels[0].tolist()):
        if score >= 0.6:
            if label != -1:
                res_label.append(label)
                res_score.append(score)
                res_box.append(box)

    return res_label, res_score, res_box

def eval(img_path, xml_path, file_handle):
    init_dict()
    # process image
    labels, scores, boxes = res(img_path)
    labels, scores, boxes = after_processing(labels, scores, boxes)

    for box, score, label in zip(boxes, scores, labels):
        #if label in labels_to_names:
        #        print(labels_to_names[label], score, box)
        if label in labels_to_names:
            names_to_number1[labels_to_names[label]]+=1

    tree = ET.parse(xml_path)
    root = tree.getroot()
    product_obj = root.findall('object')
    for obj in product_obj:
        product_name = obj.find('name')
        if product_name.text in names_to_number2:
            names_to_number2[product_name.text]+=1

    if operator.eq(names_to_number1,names_to_number2) != 1:
        for k in names_to_number1.keys():
            if(names_to_number1[k] != names_to_number2[k]):
                print(k + ": predict " + str(names_to_number1[k]) + " annotation " + str(names_to_number2[k]))
                file_handle.write(k + ": predict " + str(names_to_number1[k]) + " annotation " + str(names_to_number2[k]) + "\n")
        return False
    return True

def predict(file_path):
    ctx = {'items': []}

    for file in file_path:
        label, score, box = res(file)
        rlabel, rscore, rbox = after_processing(label, score, box)
        result = format_result(rlabel, rscore)
        ctx['items'].append(result)
    return ctx

def test(file_path):
    label, score, box = res(file_path)
    rlabel, rscore, rbox = after_processing(label, score, box)
    result = format_result(rlabel, rscore)
    print (result)
    

def plot(file_path):
    image = read_image_bgr(file_path)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image,960,1280)
    
    
    
    label, score, box = res(file_path)
    label, score, box = after_processing(label, score, box)
    for box, score, label in zip(box, score, label):
        color = label_color(label)
    
        b = box
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        print(labels_to_names[label],score)
        draw_caption(draw, b, caption)
    
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    
def eval_testset():
    total = 0
    fail_total = 0
    img_path='/home/lichunxue/Desktop/0727待学习/线上图片'
    xml_path='/home/lichunxue/Desktop/0727待学习/线上图片'
    #img_path='/home/lichunxue/Desktop/VOCjingtai/JPEGImages/'
    #xml_path='/home/lichunxue/Desktop/VOCjingtai/Annotations/'
    
    #img_path='/home/lichunxue/Desktop/VOCjingtai/testset/JPEGImages/'
    #xml_path='/home/lichunxue/Desktop/VOCjingtai/testset/Annotations/'
    
    fail_to_path = '/home/lichunxue/Desktop/0727待学习__091030h5_testfail/'
    result_file = '/home/lichunxue/Desktop/0727待学习_091030h5_testfail/result.txt'
    if os.path.exists(fail_to_path):
        shutil.rmtree(fail_to_path)
    os.mkdir(fail_to_path)
    File=open(result_file,'w+')
    File.close()
    File=open(result_file,'a')
    
    file_list = list_dir(xml_path,[],'.xml')
    #file_list.sort(key=lambda x:int(os.path.split(x)[1][:-4]))
    #file_list = file_list[135001:]
    t_start = time.time()
    for f in file_list:
        if total != 0 and total % 100 == 0:
            print('------total: {} fail {} pass rate: {}------'.format(total, fail_total, (total - fail_total) / total))
            File.close()
            File=open(result_file, 'a')
        name = os.path.split(f)[1].split('.')[0]+'.jpg'
        temp_path = os.path.join(img_path,name)
        if os.path.isfile(temp_path):
            total+=1
            if not eval(temp_path,f,File):
                fail_total += 1
                print(temp_path+"\n")
                File.write(temp_path+"\n\n")
                shutil.copy2(f, fail_to_path)
                shutil.copy2(temp_path, fail_to_path)
            else:
                #os.remove(os.path.join('/home/lichunxue/Desktop/newspace2018/testresult/0831_inference_33h5_1_375871/testfail/', os.path.split(f)[1].split('.')[0]+'.jpg'))
                #os.remove(os.path.join('/home/lichunxue/Desktop/newspace2018/testresult/0831_inference_33h5_1_375871/testfail/', os.path.split(f)[1].split('.')[0]+'.xml'))
                pass
    t_end = time.time()
    result_str = '------total: {} fail {} pass rate: {}------'.format(total, fail_total, (total - fail_total) / total)
    print(result_str)
    print('total time: {}, average time: {}/image'.format(t_end - t_start, (t_end - t_start) / len(file_list)))
    File.write('\n\n'+result_str)
    File.close()
    

if __name__ == '__main__':
    #plot("/home/lichunxue/Desktop/VOCjingtai/testset/JPEGImages/363904.jpg")
    eval_testset()
    
