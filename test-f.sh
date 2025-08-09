# !/bin/bash
# 224 320 384 448 520

# database NTIREVideoValidation 是因为Dev阶段val没有分数 因此Test修改路径即可
# 修改test_NTIRE_video_f.py 51行的videos_dir为 网盘中提供的抽帧的文件路径 即可
CUDA_VISIBLE_DEVICES=0 python -u test_NTIRE_video_f.py \
  --database NTIREVideoValidation \
  --model_name UVQA \
  --save_file results/UVQA-test.csv \
  --n_fragment 12 \
  --resize 384 \
  --crop_size 384 \
  --spatial_sample_rate 2 \
  --pretrained_path ./ckpts_save/UVQA_0.8967.pth
