import os, argparse
import numpy as np
import torch
from torchvision import transforms
import cv2

import models.UIQA as UIQA
from PIL import Image

from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
import random

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE

def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    aligned = video.shape[1]
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_upsample:

        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)



    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="Authentic Video Quality Assessment")
    parser.add_argument('--video_path', type=str, help='single video path or test csv file')
    parser.add_argument('--video_dir', type=str, help='video_dir')
    parser.add_argument('--model', type=str, default='UVQA')
    parser.add_argument('--model_path', help='Path of model snapshot.', type=str)
    parser.add_argument('--resize', type=int,default=384)
    parser.add_argument('--crop_size', help='crop_size.',type=int,default=384)
    parser.add_argument('--gpu_ids', type=list, default=None)


    args = parser.parse_args()

    return args

def extract_frames(video_path, video_length_min=8):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    images_list = []
    video_read_index = 0

    frame_idx = 0

    if video_frame_rate !=0:
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate / 2)):
                    # read_frame = cv2.resize(frame, dim)
                    read_frame = frame
                    images_list.append(read_frame)
                    video_read_index += 1
                frame_idx += 1
    else:
        # to avoid the situation that the frame rate is less than 1 fps
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length):
                    read_frame = frame
                    # read_frame = cv2.resize(frame, dim)
                    images_list.append(read_frame)
                    video_read_index += 1
            
    # if the length of read frame is less than video_length_min, copy the last frame
    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            images_list.append(read_frame)

    images_list = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_list]
    return images_list



def test_single_video(video_path, model, device, transform_asethetics, transform_distortion_preprocessing, transform_distortion):
    images_list = extract_frames(video_path)
    video_length_read, spatial_sample_rate = 8, 2
    video_aesthetics = torch.zeros([int(video_length_read/spatial_sample_rate), 3, 384, 384])             
    distortion_list = [] 
    for i in range(int(video_length_read/spatial_sample_rate)):
        image = images_list[i*spatial_sample_rate]
        image = image.convert('RGB')
        img_aesthetics = transform_asethetics(image)
        img_distortion = transform_distortion_preprocessing(image)
        video_aesthetics[i] = img_aesthetics
        distortion_list.append(img_distortion)
    distortion = torch.stack(distortion_list, dim=1)
    distortion = get_spatial_fragments(distortion,12, 12)
    distortion = distortion.permute(1,0,2,3) 
    video_distortion = torch.stack([transform_distortion(frame) for frame in distortion],dim=0)
    video_aesthetics = video_aesthetics.unsqueeze(0).to(device)
    video_distortion = video_distortion.unsqueeze(0).to(device)
    score = model(video_aesthetics, video_distortion)

    return score.item()


def test_videos(video_path, video_dir, model, device, transform_asethetics, transform_distortion_preprocessing, transform_distortion):
    column_names = ['filename','score']
    dataInfo = pd.read_csv(video_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
    video_name_list = dataInfo['filename'].tolist()
    test_data = []
    for vn in video_name_list:
        vp = os.path.join(video_dir, vn)
        score = test_single_video(vp, model, device, transform_asethetics, transform_distortion_preprocessing, transform_distortion)
        test_data.append([vn, score])
        print(vn, score)
        
    column_names = ['filename','score']
    test_data_df = pd.DataFrame(test_data, columns = column_names)
    test_data_df.to_csv('./results/UVQA-test-videos.csv', index = False)


if __name__ == '__main__':

    random_seed = 8
    torch.manual_seed(random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = parse_args()

    model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == 'UVQA':
        model = UIQA.UVQA_Model()

    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    
    transform_asethetics = transforms.Compose([transforms.Resize(args.resize),
                                               transforms.CenterCrop(args.crop_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])

    if 'csv' in args.video_path:
        test_videos(args.video_path, args.video_dir, model, device, transform_asethetics, transform_distortion_preprocessing, transform_distortion)
    else:
        score = test_single_video(args.video_path, model, device, transform_asethetics, transform_distortion_preprocessing, transform_distortion)
        print(args.video_path, score)
