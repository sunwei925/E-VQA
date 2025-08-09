import os
import cv2
import scipy.io as scio
import pandas as pd

def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    print(filename)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap=cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames

    print(video_height)
    print(video_width)

    video_read_index = 0

    frame_idx = 0
    
    video_length_min = 8

    if video_frame_rate !=0:
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % video_frame_rate == 0):
                    read_frame = frame
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                            '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                    video_read_index += 1
                frame_idx += 1
                
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(i) + '.png'), read_frame)
    else:
        # to avoid the situation that the frame rate is less than 1 fps
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length):
                    read_frame = frame
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                            '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                    video_read_index += 1
                
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(i) + '.png'), read_frame)        


    return
            
def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return

# videos_dir = '/data/sunwei_data/ntire_video'
# filename_path = 'data/train_data.csv'
# filename_path = '/mnt/sda/fk/data/VQA/KVQ/train_data.csv'
videos_dir = '/data/sunwei_data/NTIRE25_5W2_Videos/'
filename_path = '/data/sunwei_data/NTIRE25_5W2_Videos/train_data.csv'
save_folder = '/data/sunwei_data/NTIRE25_5W2_Videos/N_image_original'
os.makedirs(save_folder,exist_ok=True)

# column_names = ['filename','score']
column_names = ['filename']
dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
video = dataInfo['filename'].tolist()

video_names = []
for video_i in video:
    video_names.append(video_i)

n_video = len(video_names)






# videos_dir = '/data/sunwei_data/ntire_video'
# filename_path = 'data/val_data_nolabel.csv'

# column_names = ['filename']
# dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
# video = dataInfo['filename'].tolist()

# video_names = []
# for video_i in video:
#     video_names.append(video_i)

# n_video = len(video_names)

# videos_dir = '/data/sunwei_data/ntire_video'
# filename_path = 'data/test_data.csv'

# column_names = ['filename', 'score']
# dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
# video = dataInfo['filename'].tolist()

# video_names = []
# for video_i in video:
#     video_names.append(video_i)

# n_video = len(video_names)








# save_folder = '/mnt/sda/fk/data/VQA/KVQ/image_original'
# save_folder = '/mnt/sda/fk/data/VQA/KVQ/test_image_original'
# save_folder = '/data/sunwei_data/ntire_video/test_image_original'
for i in range(n_video):
    video_name = video_names[i]
    print('start extract {}th video: {}'.format(i, video_name))
    extract_frame(videos_dir, video_name, save_folder)
