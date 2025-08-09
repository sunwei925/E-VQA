import os 
import pandas as pd
import random
import subprocess

# idx = int(name.split('-')[-1][:-4]) maxlenth 509

QP_LIST = [[16,23],[24,31],[32,35],[36,39],[40,43],[44,47]]

def sample_videos():
    input_csv = 'RQ_VQA_results_4w_videos.csv'
    df = pd.read_csv(input_csv)
    name_list = df['filename'].tolist()
    print(len(name_list))
    sample_list = random.sample(name_list,2000)
    df_o = pd.DataFrame({'filename':sample_list})
    df_o.to_csv('./sample_videos.csv',index=False)
    # max_id = 0
    # for name in name_list:
    #     idx = int(name.split('-')[-1][:-4])
    #     if idx==509:
    #         print(name)
    #     max_id = idx if idx> max_id else max_id 

    # print(max_id)

def compress_videos():
    sample_videos = pd.read_csv('./sample_videos.csv')['filename'].tolist()
    print(len(sample_videos),sample_videos[0])
    input_video_dir = '/videos_4w/'
    save_video_dir = '/videos_compress/'
    os.makedirs(save_video_dir,exist_ok=True)
    for video_name in sample_videos:
        for qprange in QP_LIST:
            qpr = range(qprange[0],qprange[1]+1)
            qp = random.sample(qpr,1)[0]
            video_path = os.path.join(input_video_dir,video_name)
            # idx = int(video_name.split('-')[-1][:-4])
            # prefix = video_name.split('-')[0]
            idx = int(video_name[12:17])
            prefix = video_name[:11]
            save_name = prefix+'-'+str(idx+qp*1000).zfill(5)+'.mp4'
            save_path = os.path.join(save_video_dir,save_name)
            cmd = 'ffmpeg -i {} -c:v libx264 -qp {} {}'.format(video_path, qp, save_path)
            subprocess.call(cmd,shell=True)

def main():
    # sample_videos()
    compress_videos()


if __name__ == "__main__":
    main()
