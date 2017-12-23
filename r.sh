#!/bin/bash

source ./chgpath

export LD_LIBRARY_PATH=/usr/local/cuda-8.0-cudnn6.0/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/home/xyz/code1/caffe-ssd/distribute/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/xyz/code1/caffe-ssd/distribute/python:$PYTHONPATH
echo "PYTHONPATH:"$PYTHONPATH





function D2(){
python caffe_to_tensorflow.py \
--model_name=ssd_640_vgg \
--num_classes=2 \
--caffemodel_path=./tt/s3fd/SFD_fast_v2.caffemodel > tt/tmp640.txt 2>&1
}




D2






