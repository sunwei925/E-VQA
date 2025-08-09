import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import random
import cv2




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



class AVA_dataloader_pair(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, seed):
        self.database = database


        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_num'].to_list()
        image_score = np.zeros([len(image_name)])
        for i_vote in range(1,11):
            image_score += i_vote * tmp_df['vote_'+str(i_vote)].to_numpy()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)



        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):

        index_second = random.randint(0, self.length - 1)
        if index == index_second:
            index_second = (index_second + 1) % self.length
        while self.Y_train[index] == self.Y_train[index_second]:
            index_second = random.randint(0, self.length - 1)
            if index == index_second:
                index_second = (index_second + 1) % self.length

        path = os.path.join(self.data_dir,self.X_train[index])
        path_second = os.path.join(self.data_dir,self.X_train[index_second])

        img = Image.open(path)
        img = img.convert('RGB')


        img_second = Image.open(path_second)
        img_second = img_second.convert('RGB')

        img_overall = self.transform(img)
        img_second_overall = self.transform(img_second)

        y_mos = self.Y_train[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))


        y_mos_second = self.Y_train[index_second]
        y_label_second = torch.FloatTensor(np.array(float(y_mos_second)))

        return img_overall, y_label, img_second_overall, y_label_second




class AVA_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, seed):
        self.database = database


        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_num'].to_list()
        image_score = np.zeros([len(image_name)])
        for i_vote in range(1,11):
            image_score += i_vote * tmp_df['vote_'+str(i_vote)].to_numpy()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [str(image_name[i])+'.jpg' for i in index_subset]
            self.Y_train = [image_score[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)



        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):

        path = os.path.join(self.data_dir,self.X_train[index])

        img = Image.open(path)
        img = img.convert('RGB')

        img_overall = self.transform(img)

        y_mos = self.Y_train[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img_overall, y_label


    def __len__(self):
        return self.length

class UIQA_dataloader_pair(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, n_fragment=12, salient_patch_dimension=448, seed=0):
        self.database = database
        self.salient_patch_dimension = salient_patch_dimension
        self.n_fragment = n_fragment

        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_name'].to_list()
        mos = tmp_df['quality_mos'].to_list()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'all' in database:
            index_subset = index_rd
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)









        self.data_dir = data_dir                
        self.transform_aesthetics = transform
        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])
        self.transform_saliency = transforms.Compose([
            transforms.CenterCrop(self.salient_patch_dimension),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = len(self.X_train)

    def __getitem__(self, index):

        index_second = random.randint(0, self.length - 1)
        if index == index_second:
            index_second = (index_second + 1) % self.length
        while self.Y_train[index] == self.Y_train[index_second]:
            index_second = random.randint(0, self.length - 1)
            if index == index_second:
                index_second = (index_second + 1) % self.length

        path = os.path.join(self.data_dir, self.X_train[index])
        path_second = os.path.join(self.data_dir, self.X_train[index_second])

        img = Image.open(path)
        img = img.convert('RGB')


        img_second = Image.open(path_second)
        img_second = img_second.convert('RGB')

        img_aesthetics = self.transform_aesthetics(img)
        img_second_aesthetics = self.transform_aesthetics(img_second)

        img_saliency = self.transform_saliency(img)
        img_second_saliency = self.transform_saliency(img_second)


        img_distortion = self.transform_distortion_preprocessing(img)
        img_second_distortion = self.transform_distortion_preprocessing(img_second)

        img_distortion = img_distortion.unsqueeze(1)
        img_second_distortion = img_second_distortion.unsqueeze(1)

        img_distortion = get_spatial_fragments(
            img_distortion,
            fragments_h=self.n_fragment,
            fragments_w=self.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )
        img_second_distortion = get_spatial_fragments(
            img_second_distortion,
            fragments_h=self.n_fragment,
            fragments_w=self.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )

        img_distortion = img_distortion.squeeze(1)
        img_second_distortion = img_second_distortion.squeeze(1)

        img_distortion = self.transform_distortion(img_distortion)
        img_second_distortion = self.transform_distortion(img_second_distortion)

        y_mos = self.Y_train[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))


        y_mos_second = self.Y_train[index_second]

        y_label_second = torch.FloatTensor(np.array(float(y_mos_second)))


        data = {'img_aesthetics': img_aesthetics,
                'img_distortion': img_distortion,
                'img_saliency': img_saliency,
                'y_label': y_label,
                'img_second_aesthetics': img_second_aesthetics,
                'img_second_distortion': img_second_distortion,
                'img_second_saliency': img_second_saliency,
                'y_label_second': y_label_second}

        return data


    def __len__(self):
        return self.length


class UIQA_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, n_fragment=12, salient_patch_dimension=448, seed=0):
        self.database = database
        self.salient_patch_dimension = salient_patch_dimension
        self.n_fragment = n_fragment


        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['image_name'].to_list()
        mos = tmp_df['quality_mos'].to_list()

        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)

        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.8)]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.8) : ]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        elif 'all' in database:
            index_subset = index_rd
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)



        self.data_dir = data_dir
        self.transform_aesthetics = transform
        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])
        self.transform_saliency = transforms.Compose([
            transforms.CenterCrop(self.salient_patch_dimension),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = len(self.X_train)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir,self.X_train[index])

        img = Image.open(path)
        img = img.convert('RGB')

        img_aesthetics = self.transform_aesthetics(img)
        img_saliency = self.transform_saliency(img)


        img_distortion = self.transform_distortion_preprocessing(img)
        img_distortion = img_distortion.unsqueeze(1)
        img_distortion = get_spatial_fragments(
            img_distortion,
            fragments_h=self.n_fragment,
            fragments_w=self.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )
        img_distortion = img_distortion.squeeze(1)
        img_distortion = self.transform_distortion(img_distortion)

        y_mos = self.Y_train[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))

        data = {'img_aesthetics': img_aesthetics,
                'img_distortion': img_distortion,
                'img_saliency': img_saliency,
                'y_label': y_label}

        return data


    def __len__(self):
        return self.length

class KVQ_dataloader(Dataset):
    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, n_fragment=12, seed=0, spatial_sample_rate = 1):
        super(KVQ_dataloader, self).__init__() 
        if 'NTIREVideo' in database_name:
            if database_name == 'NTIREVideoValidation':
                column_names = ['filename']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = None
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                self.video_names = video_names
                self.score = score
            elif database_name == 'NTIREVideoTest':
                column_names = ['filename','score']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = None
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                self.video_names = video_names
                self.score = score
            else:                
                column_names = ['filename','score']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = dataInfo['score'].tolist()
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                if database_name == 'NTIREVideo':
                    self.video_names = video_names
                    self.score = score
                else:
                    dataInfo = pd.DataFrame(video_names)
                    dataInfo['score'] = score
                    dataInfo.columns = ['file_names', 'MOS']
                    random.seed(seed)
                    np.random.seed(seed)
                    length = 418 + 60
                    index_rd = np.random.permutation(length)
                    train_index_ref = index_rd[0:int(length * 0.8)]
                    # do not use the validation set
                    val_index_ref = index_rd[int(length * 0.8):]
                    test_index_ref = index_rd[int(length * 0.8):]
                    train_index = []
                    for i_ref in train_index_ref:
                        for i_dis in range(7):
                            train_index.append(i_ref*7+i_dis)
                    val_index = []
                    for i_ref in val_index_ref:
                        for i_dis in range(7):
                            val_index.append(i_ref*7+i_dis)
                    test_index = []
                    for i_ref in test_index_ref:
                        for i_dis in range(7):
                            test_index.append(i_ref*7+i_dis)

                    print('train_index')
                    print(train_index)
                    print('val_index')
                    print(val_index)
                    print('test_index')
                    print(test_index)
                    if database_name == 'NTIREVideo_train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()

                        self.video_names_val = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score_val = dataInfo.iloc[val_index]['MOS'].tolist()

                        self.video_names = self.video_names + self.video_names_val
                        self.score = self.score + self.score_val

                    elif database_name == 'NTIREVideo_val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name == 'NTIREVideo_test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif 'DVQ' in database_name:
            column_names = ['filename','score']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            video = dataInfo['filename'].tolist()
            score = dataInfo['score'].tolist()
            n = len(video)
            video_names = []
            for i in range(n):
                video_names.append(video[i])
            if database_name == 'DVQ':
                self.video_names = video_names 
                self.score = score 
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(n)
                train_index_ref = index_rd[0:int(n * 0.9)]
                val_index_ref = index_rd[int(n * 0.9):]
                test_index_ref = index_rd[int(n * 0.9):]
                if database_name == 'DVQ_train':
                    self.video_names = dataInfo.iloc[train_index_ref]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index_ref]['MOS'].tolist()
                elif database_name == 'DVQ_val':
                    self.video_names = dataInfo.iloc[val_index_ref]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index_ref]['MOS'].tolist()
                elif database_name == 'DVQ_test':
                    self.video_names = dataInfo.iloc[test_index_ref]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index_ref]['MOS'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.spatial_sample_rate = spatial_sample_rate

        self.n_fragment = n_fragment
        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if 'NTIREVideo' in self.database_name:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif 'DVQ' in self.database_name:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        if self.database_name == 'NTIREVideoValidation'\
            or self.database_name == 'NTIREVideoTest':
            video_score = 0
        else:
            video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       

        if 'NTIREVideo' in self.database_name:
            video_length_read = 8
        elif 'DVQ' in self.database_name:
            video_length_read = 8

        video_aesthetics = torch.zeros([int(video_length_read/self.spatial_sample_rate), video_channel, video_height_crop, video_width_crop])             
        # video_distortion = torch.zeros([int(video_length_read/self.spatial_sample_rate), video_channel, 32*self.n_fragment, 32*self.n_fragment])             
        distortion_list = []


        for i in range(int(video_length_read/self.spatial_sample_rate)):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(i*self.spatial_sample_rate)) + '.png')
            read_frame = Image.open(imge_name)
            img = read_frame.convert('RGB')
            img_distortion = self.transform_distortion_preprocessing(img)
            img_aesthetics = self.transform(img)
            video_aesthetics[i] = img_aesthetics
            distortion_list.append(img_distortion)
        distortion = torch.stack(distortion_list, dim=1)
        distortion = get_spatial_fragments(distortion,self.n_fragment,self.n_fragment)
        distortion = distortion.permute(1,0,2,3) 
        # video_distortion = self.transform_distortion(distortion)
        video_distortion = torch.stack([self.transform_distortion(frame) for frame in distortion],dim=0)

        data = {
            'img_aesthetics': video_aesthetics,
            'img_distortion': video_distortion,
            # 'img_saliency': None,
            'y_label': video_score,
            'video_name': video_name
        }

        return data
        # return transformed_video, video_score, video_name





class KVQ_dataloader_pair(Dataset):
    def __init__(self, data_dir, filename_path, transform, database_name, crop_size, n_fragment=12, seed=0, spatial_sample_rate = 1):
        super(KVQ_dataloader_pair, self).__init__() 
        if 'NTIREVideo' in database_name:
            if database_name == 'NTIREVideoValidation':
                column_names = ['filename']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = None
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                self.video_names = video_names
                self.score = score
            elif database_name == 'NTIREVideoTest':
                column_names = ['filename','score']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = None
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                self.video_names = video_names
                self.score = score
            else:                
                column_names = ['filename','score']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = dataInfo['score'].tolist()
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                if database_name == 'NTIREVideo':
                    self.video_names = video_names
                    self.score = score
                else:
                    dataInfo = pd.DataFrame(video_names)
                    dataInfo['score'] = score
                    dataInfo.columns = ['file_names', 'MOS']
                    random.seed(seed)
                    np.random.seed(seed)
                    length = 418
                    index_rd = np.random.permutation(length)
                    train_index_ref = index_rd[0:int(length * 0.8)]
                    # do not use the validation set
                    val_index_ref = index_rd[int(length * 0.8):]
                    test_index_ref = index_rd[int(length * 0.8):]
                    train_index = []
                    for i_ref in train_index_ref:
                        for i_dis in range(7):
                            train_index.append(i_ref*7+i_dis)
                    val_index = []
                    for i_ref in val_index_ref:
                        for i_dis in range(7):
                            val_index.append(i_ref*7+i_dis)
                    test_index = []
                    for i_ref in test_index_ref:
                        for i_dis in range(7):
                            test_index.append(i_ref*7+i_dis)

                    print('train_index')
                    print(train_index)
                    print('val_index')
                    print(val_index)
                    print('test_index')
                    print(test_index)
                    if database_name == 'NTIREVideo_train':
                        self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                    elif database_name == 'NTIREVideo_val':
                        self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                    elif database_name == 'NTIREVideo_test':
                        self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                        self.score = dataInfo.iloc[test_index]['MOS'].tolist()
        elif 'DVQ' in database_name:
            if not isinstance(filename_path, list):
                column_names = ['filename','score']
                dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                video = dataInfo['filename'].tolist()
                score = dataInfo['score'].tolist()
                n = len(video)
                video_names = []
                for i in range(n):
                    video_names.append(video[i])
                if database_name == 'DVQ':
                    self.video_names = video_names 
                    self.score = score 
            else:
                self.video_names = []
                self.score = []
                for fp, dd in zip(filename_path, data_dir):
                    column_names = ['filename','score']
                    dataInfo = pd.read_csv(fp, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
                    video = dataInfo['filename'].tolist()
                    score = dataInfo['score'].tolist()
                    for i in range(len(video)):
                        self.video_names.append(os.path.join(dd,video[i]))
                        self.score.append(score[i])

            # else:
            #     dataInfo = pd.DataFrame(video_names)
            #     dataInfo['score'] = score
            #     dataInfo.columns = ['file_names', 'MOS']
            #     random.seed(seed)
            #     np.random.seed(seed)
            #     index_rd = np.random.permutation(n)
            #     train_index_ref = index_rd[0:int(n * 0.9)]
            #     val_index_ref = index_rd[int(n * 0.9):]
            #     test_index_ref = index_rd[int(n * 0.9):]
            #     if database_name == 'DVQ_train':
            #         self.video_names = dataInfo.iloc[train_index_ref]['file_names'].tolist()
            #         self.score = dataInfo.iloc[train_index_ref]['MOS'].tolist()
            #     elif database_name == 'DVQ_val':
            #         self.video_names = dataInfo.iloc[val_index_ref]['file_names'].tolist()
            #         self.score = dataInfo.iloc[val_index_ref]['MOS'].tolist()
            #     elif database_name == 'DVQ_test':
            #         self.video_names = dataInfo.iloc[test_index_ref]['file_names'].tolist()
            #         self.score = dataInfo.iloc[test_index_ref]['MOS'].tolist()


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name
        self.spatial_sample_rate = spatial_sample_rate

        self.n_fragment = n_fragment
        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if 'NTIREVideo' in self.database_name:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif 'DVQ' in self.database_name:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        if self.database_name == 'NTIREVideoValidation'\
            or self.database_name == 'NTIREVideoTest':
            video_score = 0
        else:
            video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        idx_second = random.randint(0, self.length - 1)
        if idx == idx_second:
            idx_second = (idx_second + 1) % self.length
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        video_score_second = torch.FloatTensor(np.array(float(self.score[idx_second])))
        while video_score == video_score_second:
            idx_second = random.randint(0, self.length - 1)
            if idx == idx_second:
                idx_second = (idx_second + 1) % self.length
            video_score_second = torch.FloatTensor(np.array(float(self.score[idx_second])))
        video_score_second = torch.FloatTensor(np.array(float(self.score[idx_second])))
        video_name_second = self.video_names[idx_second]
        video_name_str_second = video_name_second[:-4]

        # path_name = os.path.join(self.videos_dir, video_name_str)
        # path_name_second = os.path.join(self.videos_dir, video_name_str_second)
        path_name = video_name_str
        path_name_second = video_name_str_second

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       

        if 'NTIREVideo' in self.database_name:
            video_length_read = 8
        elif 'DVQ' in self.database_name:
            video_length_read = 8

        video_aesthetics = torch.zeros([int(video_length_read/self.spatial_sample_rate), video_channel, video_height_crop, video_width_crop])             
        video_aesthetics_second = torch.zeros([int(video_length_read/self.spatial_sample_rate), video_channel, video_height_crop, video_width_crop])             
        # video_distortion = torch.zeros([int(video_length_read/self.spatial_sample_rate), video_channel, 32*self.n_fragment, 32*self.n_fragment])             
        distortion_list = []
        distortion_second_list = []


        for i in range(int(video_length_read/self.spatial_sample_rate)):
            imge_name = os.path.join(path_name, '{:03d}'.format(int(i*self.spatial_sample_rate)) + '.png')
            read_frame = Image.open(imge_name)
            img = read_frame.convert('RGB')
            img_aesthetics = self.transform(img)
            video_aesthetics[i] = img_aesthetics

            img_distortion = self.transform_distortion_preprocessing(img)
            distortion_list.append(img_distortion)

            imge_name_second = os.path.join(path_name_second, '{:03d}'.format(int(i*self.spatial_sample_rate)) + '.png')
            read_frame_second = Image.open(imge_name_second)
            img_second = read_frame_second.convert('RGB')
            img_aesthetics_second = self.transform(img_second)
            video_aesthetics_second[i] = img_aesthetics_second

            img_distortion_second = self.transform_distortion_preprocessing(img_second)
            distortion_second_list.append(img_distortion_second)

        distortion = torch.stack(distortion_list, dim=1)
        distortion = get_spatial_fragments(distortion,self.n_fragment,self.n_fragment)
        distortion = distortion.permute(1,0,2,3) 
        # video_distortion = self.transform_distortion(distortion)
        video_distortion = torch.stack([self.transform_distortion(frame) for frame in distortion],dim=0)

        distortion_second = torch.stack(distortion_second_list, dim=1)
        distortion_second = get_spatial_fragments(distortion_second,self.n_fragment,self.n_fragment)
        distortion_second = distortion_second.permute(1,0,2,3) 
        # video_distortion = self.transform_distortion(distortion)
        video_distortion_second = torch.stack([self.transform_distortion(frame) for frame in distortion_second],dim=0)
        data = {
            'img_aesthetics': video_aesthetics,
            'img_distortion': video_distortion,
            'img_second_aesthetics': video_aesthetics_second,
            'img_second_distortion': video_distortion_second,
            # 'img_saliency': None,
            'y_label': video_score,
            'y_label_second': video_score_second,
            'video_name': video_name
        }

        return data
        # return transformed_video, video_score, video_name





