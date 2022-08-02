"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2021-04-14 11:17
 * Filename      : valid_script.py
 * Description   : 
"""

import os 
import json
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# 绘制图片
class draw_pic(object):
    def __init__(self):
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

        self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]

        self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255)]

        color_list = np.array(
            [
                1.000, 1.000, 1.000,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.167, 0.000, 0.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32)
        color_list = color_list.reshape((-1, 3)) * 255

        self.num_joints = 17
        self.theme = 'white'
        colors = [(color_list[_]).astype(np.uint8)
                  for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(
            len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
        self.names = ['p']

    def draw_pic_label(self, img_fp, img_label_list, save_fp):
        img_t = cv2.imread(img_fp)
        for img_label in img_label_list:
            # pts = img_label['keypoints']
            bbox = img_label['bbox']
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            self.add_coco_bbox(img_t, bbox)
            # self.add_coco_hp(img_t,pts)
        self.save_img(img_t, save_fp)
        pass

    def save_img(self, image, save_fp):
        cv2.imwrite(save_fp, image)
        pass

    def add_coco_bbox(self, image, bbox, cat=0, conf=1, show_txt=True):
        bbox = np.array(bbox, dtype=np.int32)
        # cat = (int(cat) + 1) % 80
        cat = int(cat)
        # print('cat', cat, self.names[cat])
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        txt = '{}{:.1f}'.format(self.names[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(
            image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            cv2.rectangle(image,
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(image, txt, (bbox[0], bbox[1] - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def add_coco_hp(self, image, points):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 3)[:,:2]
        for j in range(self.num_joints):
            cv2.circle(image,(points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
        for j, e in enumerate(self.edges):
            if points[e].min() > 0:
                cv2.line(image, (points[e[0], 0], points[e[0], 1]),
                         (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                         lineType=cv2.LINE_AA)


 
def copy_imgs_to_des(img_ids_list, data_dir, des_dir):
    """ 将指定的图片数据转移到指定的目录下

    """
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for img in tqdm(img_ids_list[::50]):
        img_fp = os.path.join(data_dir, str(img))
        des_img_fp = os.path.join(des_dir, str(img))
        shutil.copy(img_fp, des_img_fp)
    pass


def mk_valid_imgs():
    this_dir = os.getcwd()
    data_dir = os.path.join(this_dir, 'data/COCO2017_VOC2012_person/val2017') 
    des_dir = os.path.join(this_dir, 'test_imgs') 

    imgs_list = os.listdir(data_dir)
    imgs_list = [img for img in imgs_list if img.endswith('jpg')]
    copy_imgs_to_des(imgs_list, data_dir, des_dir)

def draw_4_valid():

    this_dir = os.getcwd()
    # 加载标签文件，提取bbox和pts数据
    data_dir = os.path.join(this_dir, 'data/COCO2017_VOC2012_person')
    tmp_valid_img_dir = os.path.join(this_dir, 'test_imgs') 
    anno_all_fp = os.path.join(data_dir, 'annotations_all.json')
    # test_anno_fp = os.path.join(anno_dir, 'person_keypoints_val2017.json')
    # img_dir = os.path.join(data_dir, 'all_fms_imgs')

    with open(anno_all_fp,'r')as rd:
        anno_all = json.loads(rd.read())
    annos_labels = anno_all['annotations']
    images_info = anno_all['images']

    # 加载图片
    imgs = os.listdir(tmp_valid_img_dir)
    if 'ans' not in imgs:
        os.makedirs(os.path.join(tmp_valid_img_dir,'ans'))
    if 'ans_origin' not in imgs:
        os.makedirs(os.path.join(tmp_valid_img_dir,'ans_origin'))
    imgs = [img for img in imgs if img not in ['ans', 'ans_origin']]
    
    fname_with_imgid = {img:None for img in imgs}
    for image in images_info:
        if image['file_name'] in fname_with_imgid:
            fname_with_imgid[image['file_name']] = image['id']

    imgid_with_fname = {v:k for k,v in fname_with_imgid.items()}

    imgs_bbox_list = {fname_with_imgid[img]:[] for img in imgs}
    print("提取所有图片的annos......")
    for anno in annos_labels:
        img_id = anno['image_id']
        if img_id in imgs_bbox_list:
            imgs_bbox_list[img_id].append(anno)

    print('开始绘图......')
    draw_handler = draw_pic()
    for k,v in tqdm(imgs_bbox_list.items()):
        img_fp = os.path.join(tmp_valid_img_dir, imgid_with_fname[k])
        save_fp = os.path.join(tmp_valid_img_dir, 'ans',imgid_with_fname[k].split('.')[0]+"_ans.jpg")
        draw_handler.draw_pic_label(img_fp, v, save_fp)

    # for img in imgs:
        # if img not in ["ans", "ans_origin"]:
            # img_id = img.split('.')[0]        
            # img_fp = os.path.join(tmp_valid_img_dir, img)
            # img_label = annos_labels[int(img_id.strip())]
            # save_fp = os.path.join(tmp_valid_img_dir, 'ans', img_id+"_ans.jpg")
            # draw_handler.draw_pic_label(img_fp, img_label, save_fp)
    pass

if __name__ == '__main__':
    # mk_valid_imgs()
    draw_4_valid()
    pass
