"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2020-10-16 09:42
 * Filename      : merge_dataset.py
 * Description   : 将两个fms数据集（coco格式）合成一个 
"""

import os
import json
import cv2
import random
import shutil
from tqdm import tqdm
import numpy as np
# from fms_labeled_dataset_mkr import split_data_to_train_test

def mkdir_4_coco_dataset(dataset_dir):
    all_imgs_dir = os.path.join(save_dir, 'all_fms_imgs')
    train2017 = os.path.join(save_dir, 'train2017')
    val2017 = os.path.join(save_dir, 'val2017')
    if not os.path.exists(all_imgs_dir):
        os.makedirs(all_imgs_dir)
    if not os.path.exists(train2017):
        os.makedirs(train2017)
    if not os.path.exists(val2017):
        os.makedirs(val2017)
    pass

def get_anno(dataset_dir):
    anno_path = os.path.join(dataset_dir,"annotations_all.json")
    with open(anno_path, 'r')as rd:
        coco_dict = json.loads(rd.read())
    return coco_dict
    pass


def update_anno(anno_list, img_list):
    assert len(anno_list) == len(img_list)
    for i,img in enumerate(img_list):
        anno_list[i]['image_id'] = anno_list[i]['id'] = img['id']

    pass

def move_img(coco_dict, img_src_dir, img_des_dir):
    """将数据集的图片放到合并的目录下
    """
    src_imgs_list = coco_dict['images']
    anno_list = coco_dict['annotations'] 
    for img in tqdm(src_imgs_list):
        img_src = os.path.join(img_src_dir, img['file_name'])
        img_des = os.path.join(img_des_dir, img['file_name'])
        shutil.move(img_src, img_des)
    pass

def fms_data_shuffer(cnt, start_from_zero):
    """
        对于更大数据量的混洗可以参考如下方案:
            https://www.pythonheidong.com/blog/article/389773/
    """
    if start_from_zero:
        numbers = list(range(cnt)) 
    else:
        numbers = list(range(cnt+1)) 
    random.seed(0)
    random.shuffle(numbers)
    return numbers 
    pass

def copy_imgs_to_des(img_ids_list, data_dir, des_dir):
    """ 将指定的图片数据转移到指定的目录下

    """
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for img in tqdm(img_ids_list):
        # img_fp = os.path.join(data_dir, str(img)+".jpg")
        # des_img_fp = os.path.join(des_dir, str(img)+".jpg")
        img_fp = os.path.join(data_dir, img)
        des_img_fp = os.path.join(des_dir, img)
        shutil.copy(img_fp, des_img_fp)
    pass

def save_annotations(anno_list, save_fp, coco_dict, images_info):
    """该函数在保存train/val的image信息时，仍旧依赖anno:image == 1:1的关系
    """
    json_dict = {'images': [], 'type': 'instance', 'annotations': [],'categories': [{'supercategory':'none', 'id':1, 'name':'person'}]}
    json_dict['images'] = images_info # 保存images信息
    annos = coco_dict['annotations']
    # 提取annos信息并保存
    for anno in tqdm(annos):
        if anno['image_id'] in anno_list:
            json_dict['annotations'].append(anno)
    parent_dir = os.path.dirname(save_fp)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_fp, 'w')as wr:
        wr.write(json.dumps(json_dict))
    pass

def split_data_to_train_test(data_dir, cnt, coco_dict, start_from_zero=False):
    """将数据集图片混洗，并按照8:2比例分成训练和测试数据
    
    Args:
        data_dir: 存放所有图片的目录
        cnt: 图片的总数
    """
    all_images_list = coco_dict['images']
    img_ids_list = fms_data_shuffer(cnt, start_from_zero=start_from_zero) 
    train_rate=0.8
    cnt = len(img_ids_list)
    train_nums = int(cnt * train_rate)
    # 提取训练集的image信息
    train_idx_list = []
    train_img_info = []
    train_img_fp = []
    for idx in img_ids_list[:train_nums]:
        train_img_info.append(all_images_list[idx])
        train_idx_list.append(all_images_list[idx]['id'])
        train_img_fp.append(all_images_list[idx]['file_name'])
    # 提取验证集的image信息 
    val_idx_list = []
    val_img_info = []
    val_img_fp = []
    for idx in img_ids_list[train_nums:]:
        val_img_info.append(all_images_list[idx])
        val_idx_list.append(all_images_list[idx]['id'])
        val_img_fp.append(all_images_list[idx]['file_name'])

    save_data_dir = os.path.dirname(data_dir) # 合并后数据集的主目录
    anno_save_dir = os.path.join(save_data_dir, 'annotations')
    # 保存train图片和annotation
    train_data_dir = os.path.join(save_data_dir, 'train2017')
    print("start coping train imgs to des_dir......")
    copy_imgs_to_des(train_img_fp,data_dir, train_data_dir)
    print("start making train annos and save to des_dir......")
    save_annotations(train_idx_list, os.path.join(anno_save_dir, 'person_instance_train2017.json'), coco_dict, train_img_info)
    print('finish train data making ......')
    # 保存test图片和annotation
    test_data_dir = os.path.join(save_data_dir, 'val2017')
    print("start coping test imgs to des_dir......")
    copy_imgs_to_des(val_img_fp, data_dir, test_data_dir)
    print("start making test annos and save to des_dir......")
    save_annotations(val_idx_list, os.path.join(anno_save_dir, 'person_instance_val2017.json'),coco_dict, val_img_info)
    print('finish test data making ......')
    pass    

def merge_coco_dict(coco_dict_1, coco_dict_2):
    coco_dict_1['images'].extend(coco_dict_2['images'])
    coco_dict_1['annotations'].extend(coco_dict_2['annotations'])
    return coco_dict_1

def save_annos(coco_dict, save_dir):
    anno_fp = os.path.join(save_dir, 'annotations_all.json')
    with open(anno_fp, 'w')as wr:
        wr.write(json.dumps(coco_dict))

    pass

def merge_dataset(dataset_fp_1, dataset_fp_2, save_dir):
    """将两个fms数据集合并成一个
    """
    # 构建coco数据集的形式的目录结构
    mkdir_4_coco_dataset(save_dir)
    
    # 提取label数据
    coco_dict_1 = get_anno(dataset_fp_1)
    coco_dict_2 = get_anno(dataset_fp_2)

    src_dir_1 = os.path.join(dataset_fp_1, 'all_fms_imgs')
    src_dir_2 = os.path.join(dataset_fp_2, 'all_fms_imgs')
    des_dir = os.path.join(save_dir, 'all_fms_imgs')
    # start_cnt = len(coco_dict_1['images'])
    # 合并图片
    # move_img(coco_dict_1, src_dir_1, des_dir)
    move_img(coco_dict_2, src_dir_2, des_dir)
    # 合并label
    coco_dict_merge = merge_coco_dict(coco_dict_1, coco_dict_2)    
    save_annos(coco_dict_merge, save_dir) 
    pass

def mk_train_val_dataset(dataset_dir):
    """制作训练和测试集
        不过这里引用了fms_labeled_dataset_mkr中的split_data_to_train_test函数，需要注意的是，其调用的fms_data_shuffer函数中cnt+1要改成cnt
    """
    all_fms_imgs = os.path.join(dataset_dir, 'all_fms_imgs')
    coco_dict = get_anno(dataset_dir)
    cnt = len(coco_dict['images'])
    split_data_to_train_test(all_fms_imgs, cnt, coco_dict, start_from_zero=True)
    pass

if __name__ == "__main__":
    # # ========================================= ops-1 ===============================================
    this_dir = os.getcwd()
    data_dir = os.path.join(this_dir, 'data')
    
    # ================== 第一次merge ==================
    # fms_data1_dir = os.path.join(data_dir, 'COCO2017_train') 
    # fms_data2_dir = os.path.join(data_dir, 'COCO2017_val') 
    # save_dir = os.path.join(data_dir, 'COCO2017_person') 
    # merge_dataset(fms_data1_dir, fms_data2_dir, save_dir)

    # ================== 第二次merge ==================
    # fms_data1_dir = os.path.join(data_dir, 'COCO2017_person') 
    # fms_data2_dir = os.path.join(data_dir, 'voc2coco') 
    # save_dir = os.path.join(data_dir, 'COCO2017_VOC2012_person') 
    # merge_dataset(fms_data1_dir, fms_data2_dir, save_dir)


    # 开始制作train/val
    save_dir = os.path.join(data_dir, 'COCO2017_VOC2012_person') 
    mk_train_val_dataset(save_dir)
    # pass
