#!/bin/bash 

python extract_frames.py --dataset NTIREVideoTrain --videos_dir ./videos_compress/ --dataset_file ./videos_compress_results.csv --save_folder image_compress_original --resize 384 --bid $1 --eid $2
