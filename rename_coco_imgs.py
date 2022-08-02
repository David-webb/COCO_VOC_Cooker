"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2021-04-14 10:45
 * Filename      : rename_coco_imgs.py
 * Description   : 
"""
import os
import shutil
this_dir = os.getcwd() 
data_dir = os.path.join(this_dir, 'data/COCO2017_VOC2012_person/all_fms_imgs') 
imgs_names = os.listdir(data_dir)
for img in imgs_names:
    # if img.startswith('000000'):
    if len(img.split('.')[0]) <= 6:
        src_ = os.path.join(data_dir, img)
        idx = '000000'+img.split('.')[0]
        des_ = os.path.join(data_dir, str(idx)+'.jpg')
        shutil.move(src_, des_)
