import os
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2
from torchvision import transforms
import csv


class VideoDataset_images_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D ,filename_path, transform, database_name, crop_size, feature_type, is_train=True):
        super(VideoDataset_images_with_motion_features, self).__init__()
        self.is_train = is_train
        if database_name == 'NTIRE':
            # 读取csv
            video_names = []
            scores= []
            with open(filename_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                # 读取表头（如果有）
                headers = next(reader, None)
                if headers:
                    print("表头:", headers)
                # 遍历每一行
                for row in reader:
                    if is_train:
                        video_name, score = row[0], row[1]
                    else:
                        video_name, score = row[0], 0.0
                    video_names.append(video_name)
                    scores.append(score)
            self.video_names = video_names
            self.score = scores

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'NTIRE':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        
        # breakpoint()
        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3
        video_length_read = 20
        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             

        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # breakpoint()
        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i   
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048+256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

       
        return transformed_video, transformed_feature, video_score, video_name


if __name__ == '__main__':
    videos_dir = 'videos_dir'
    feature_dir = 'feature_dir'
    datainfo_train = 'datainfo_train'
    is_train = True

    transformations_train = transforms.Compose([transforms.Resize(520), transforms.RandomCrop(448), transforms.ToTensor(),\
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    trainset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_train, transformations_train, 
                                                    'NTIRE', 448, 'SlowFast', is_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=2,
        shuffle=False, num_workers=0)

    for data in train_loader:
        transformed_video, transformed_feature, video_score, video_name = data
        breakpoint()
