"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2021-04-13 13:50
 * Filename      : coco_extractor.py
 * Description   : 从coco目标检测数据集中提取出想要的类别数据 
"""
import os
import json
from tqdm import tqdm
import shutil

# coco类别大全(共80类，最大编号90:在编号的时候,有些数字跳过了)
categories = [
    {'supercategory': 'person', 'id': 1, 'name': 'person'},
    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
    {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
    {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
    {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
    {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
    {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
    {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
    {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
    {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
    {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
    {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
    {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
    {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
    {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
    {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
    {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
    {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
    {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
    {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
    {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
    {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
    {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
    {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
    {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
    {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
    {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
    {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
    {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
    {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
    {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
    {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
    {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
    {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
    {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
    {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
    {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
    {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
    {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
    {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
    {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
    {'supercategory': 'food', 'id': 52, 'name': 'banana'},
    {'supercategory': 'food', 'id': 53, 'name': 'apple'},
    {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
    {'supercategory': 'food', 'id': 55, 'name': 'orange'},
    {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
    {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
    {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
    {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
    {'supercategory': 'food', 'id': 60, 'name': 'donut'},
    {'supercategory': 'food', 'id': 61, 'name': 'cake'},
    {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
    {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
    {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
    {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
    {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
    {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
    {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
    {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
    {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
    {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
    {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
    {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
    {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
    {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
    {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
    {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
    {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
    {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
    {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
    {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
    {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
    {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
    {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
    {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'},
]

cat_with_id = {k['name']:k['id'] for k in categories}
anno_with_cat = {k['name']:k for k in categories}

def catProj2Id(cat_name):
    """将catgory名称映射到Id
    """
    global cat_with_id
    if cat_name not in cat_with_id.keys():
        print("wrong cat name:%s! check and try again!" % cat_name)
        return False
    return cat_with_id[cat_name]
    pass

def wrap_coco_json(categories_list):
    """根据categories_list构建coco_dict文件框架
    """
    coco_json = {'images': [], 'type': 'instance', 'annotations': [],'categories': []}

    global anno_with_cat 
    for cat_name in categories_list:
        if cat_name not in anno_with_cat.keys():
            print("wrong cat name:%s! check and try again!" % cat_name)
            return False
        coco_json['categories'].append(anno_with_cat[cat_name])     
    return coco_json
    pass

def read_coco_json(json_fp):
    with open(json_fp,'r')as rd:
        coco_label_dic = json.loads(rd.read())
    return coco_label_dic


def copy_imgs_to_des(img_ids_list, data_dir, des_dir):
    """ 将指定位置的图片复制到指定目录下
    """
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for img in tqdm(img_ids_list):
        img_fp = os.path.join(data_dir, img) # str(img)+".jpg"
        des_img_fp = os.path.join(des_dir, img) # str(img)+".jpg"
        shutil.copy(img_fp, des_img_fp)
    pass

def save_coco_label_dic(coco_json, save_dir):
    save_fp = os.path.join(save_dir, "annotations_all.json")
    with open(save_fp, 'w')as wr:
        wr.write(json.dumps(coco_json))

def imgid_with_imginfo(image_info):
    """
        image_info = [
            {
                "license":xxx,
                "file_name": xxx,
                'coco_url': xxx,
                'height':xxx,
                'width': xxx,
                "date_captured": xxxx,
                "flickr_url":xxxxm
                "id":xxx,
            },
            .....
        ]
    """
    ans_info = {}
    for img_info in image_info:
        ans_info[img_info['id']] = {
                "file_name":img_info["file_name"],
                'height':img_info["height"],
                'width':img_info["width"],
                "id":img_info["id"],
                }
    return ans_info

def get_img_list_from_label_json(json_fp):
    coco_json = read_coco_json(json_fp)
    imgs_info = coco_json['images']
    return [img['file_name'] for img in imgs_info]
    pass

def just_copy_imgs(json_fp, images_dir, save_dir):
    imgs_list = get_img_list_from_label_json(json_fp)
    copy_imgs_to_des(imgs_list, images_dir, os.path.join(save_dir,'all_fms_imgs')) 
    pass


def extract_specify_cls_from_coco(categories_list, json_fp, images_dir, save_dir):
    """
    Args:
        categories_list: 需要提取的类别列表
        json_fp: coco类型数据集的json文件
        images_dir: json_fp对应的图片目录
        save_dir: 提取结果的保存目录
    """
    # 数据保存路径检查 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 构建基础coco_label_dict
    coco_json = wrap_coco_json(categories_list)
    # 读取coco的label文件
    coco_label_dic = read_coco_json(json_fp)
    annos = coco_label_dic['annotations']
    images = coco_label_dic['images']
    imgs_info = imgid_with_imginfo(images)
    cat_ids = [catProj2Id(c) for c in categories_list] 
    img_list = []
    saved_imgs_ids = []
    print("提取label数据......")
    for anno in tqdm(annos):
        # print(anno['image_id'], images[i]['id'])
        # break
        if anno['category_id'] in cat_ids:
            coco_json['annotations'].append(anno)
            # print(i)
            if anno['image_id'] not in saved_imgs_ids:
                saved_imgs_ids.append(anno['image_id'])
                tmp_img_info = imgs_info[anno['image_id']]
                coco_json['images'].append(tmp_img_info)
                img_list.append(tmp_img_info['file_name'])       # 保存图片名称
                # print(tmp_img_info['file_name'])

    save_coco_label_dic(coco_json, save_dir) 
    print('提取图片数据')
    copy_imgs_to_des(img_list, images_dir, os.path.join(save_dir,'all_fms_imgs')) 
    pass

if __name__ == "__main__":
    this_dir = os.getcwd()
    data_dir = os.path.join(os.path.dirname(this_dir), 'data')

    # cat_list = ['person'] 
    # json_fp = "/mnt/share/COCO/annotations/instances_train2017.json"
    # images_dir = "/mnt/share/COCO/images/train2017"
    # save_dir = os.path.join(data_dir, 'COCO2017_train') 
    # extract_specify_cls_from_coco(cat_list, json_fp, images_dir, save_dir)
    # # ============这里是因为除了错，在保存完label后，单独保存了一下图片================
    # # json_fp = os.path.join(save_dir, "annotations_all.json")
    # # just_copy_imgs(json_fp, images_dir, save_dir)
    # # =================================================================================
    cat_list = ['person'] 
    json_fp = "/mnt/share/COCO/annotations/instances_val2017.json"
    images_dir = "/mnt/share/COCO/images/val2017"
    save_dir = os.path.join(data_dir, 'COCO2017_val') 
    extract_specify_cls_from_coco(cat_list, json_fp, images_dir, save_dir)
    pass

