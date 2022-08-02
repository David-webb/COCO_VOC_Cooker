# 该项目用于将从COCO和VOC数据集中提取指定类别的目标检测数据，并支持将VOC转成COCO后进行合并,以及对合并后的数据集进行绘图验证
## COCO_extractor/coco_extractor.py 
从COCO数据集中提取指定类别的目标检测数据

## VOC_extracter/extractor.py
从VOC数据集中提取指定类别的目标检测数据

##voc2coco.py
将VOC数据集转换成COCO形式保存

## merge_dataset.py
将两个COCO形式的数据集进行合并。这里要注意anno和image是多对1的关系，而在FMS项目中刚好是1对1的关系(fms场景都是单目标姿态识别)，所以FMS项目中数据集制作的代码不能直接拿来即用。

## valid_script.py
从验证数据集中抽取一部分作为验证数据集正确性的样本，并提供了绘图函数

## data/
该目录保存所有用到的数据集

## test_imgs/
该目录保存用于验证正确性的图片

