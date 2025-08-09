# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import torch

import random

#from data_loader import VideoDataset_images_motion_features
import IQADataset
import models.UIQA as UIQA
# from model import baseline_Swin_motion
# import baseline_Swin_motion

from torchvision import transforms


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    if config.model_name == 'Model_SwinT_LSVQ':
        model = baseline_Swin_motion.Model_SwinT_LSVQ()
        print('The current model is ' + config.model_name)
    if config.model_name == 'UVQA':
        model = UIQA.UVQA_Model()
        print('The current model is ' + config.model_name)
    elif config.model_name == 'Model_MobileNet_V2_LSVQ':
        print('The current model is ' + config.model_name)
        model = baseline_Swin_motion.Model_MobileNet_V2_LSVQ()

    
    # model.load_state_dict(torch.load(config.pretrained_path))
    # model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # load model
    model.load_state_dict(torch.load(config.pretrained_path))
    model.to(device)
    
    # transformations_test = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
    #     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    transforms_test = transforms.Compose([transforms.Resize(config.resize),
                                            transforms.CenterCrop(config.crop_size), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    ## training data
    if config.database == 'NTIREVideoValidation':
        datainfo = './test_data.csv'
        # please change the dir
        videos_dir = '/data/sunwei_data/NTIRE25_5W2_Videos/test_image_origina'

        # testset = VideoDataset_images_motion_features(videos_dir, datainfo, transformations_test, 'NTIREVideoValidation', config.crop_size, 'SlowFast_Fast_sr1', seed=8, spatial_sample_rate=config.spatial_sample_rate)
        testset = IQADataset.KVQ_dataloader(videos_dir,
                                              datainfo,
                                              transforms_test,
                                              'NTIREVideoValidation',
                                              config.crop_size,
                                              config.n_fragment,
                                              seed=8,
                                              spatial_sample_rate=config.spatial_sample_rate)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    # # Test
    with torch.no_grad():
        model.eval()
        y_output = []
        video_names = []
        test_data = []
        for i, data in enumerate(test_loader):
            img_aesthetics = data['img_aesthetics'].to(device)
            img_distortion = data['img_distortion'].to(device)
            video_name = data['video_name']
            
            # feature_3D = feature_3D.to(device)
            outputs = model(img_aesthetics, img_distortion)

            y_output.append(outputs.item())
            video_names.append(video_name[0])
            print(video_name[0])
            test_data.append([video_name[0], outputs.item()])
        
    column_names = ['filename','score']
    test_data_df = pd.DataFrame(test_data, columns = column_names)
    test_data_df.to_csv(config.save_file, index = False)
    
            


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--spatial_sample_rate', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=8)
    parser.add_argument('--n_fragment', type=int, default=12)
    
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    main(config)

# CUDA_VISIBLE_DEVICES=3 python -u test_NTIRE_video.py \
# --database NTIREVideoValidation \
# --model_name Model_SwinT_LSVQ \
# --save_file results/Model_SwinT_LSVQ_spatial_sample_rate_2.csv \
# --resize 384 \
# --crop_size 384 \
# --spatial_sample_rate 2 \
# --pretrained_path ckpts/Model_SwinT_LSVQ_NTIREVideo_plcc_NR_v0_epoch_22_SRCC_0.839951.pth
