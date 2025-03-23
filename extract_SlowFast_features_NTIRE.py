# -*- coding: utf-8 -*-
import argparse
import os
from torch.utils import data
import numpy as np
import torch
import torch.nn as nn
import csv
from pytorchvideo.models.hub import slowfast_r50
from torchvision import transforms
import cv2
from PIL import Image


class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize, database_name):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()

        if database_name == 'NTIRE':
            # 读取csv
            video_names = []
            with open(filename_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                # 读取表头（如果有）
                headers = next(reader, None)
                if headers:
                    print("表头:", headers)
                # 遍历每一行
                for row in reader:
                    video_name = row[0]
                    video_names.append(video_name)
            self.video_names = video_names

        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.database_name = database_name
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # breakpoint()
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        filename=os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
           video_clip = 10
        else:
            video_clip = int(video_length/video_frame_rate)

        video_clip_min = 20

        video_length_clip = 32             

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_name_str


def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list

class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            
        return slow_feature, fast_feature

def main(config):
    # breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = slowfast()
    model = model.to(device)
    resize = config.resize

    videos_dir = config.config
    datainfo = config.datainfo
    feature_save_folder = config.feature_save_folder

    transformations = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])
    trainset = VideoDataset_NR_SlowFast_feature(videos_dir, datainfo, transformations, resize, 'NTIRE')
    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    # do validation after each epoch
    with torch.no_grad():
        model.eval()
        for i, (video, video_name) in enumerate(train_loader):
            # breakpoint()
            video_name = video_name[0]
            print(video_name)
            if not os.path.exists(feature_save_folder + video_name):
                os.makedirs(feature_save_folder + video_name)
            for idx, ele in enumerate(video):
                # ele = ele.to(device)
                ele = ele.permute(0, 2, 1, 3, 4)             
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)
                np.save(os.path.join(feature_save_folder, video_name, 'feature_' + str(idx) + '_slow_feature'), slow_feature.to('cpu').numpy())
                np.save(os.path.join(feature_save_folder, video_name, 'feature_' + str(idx) + '_fast_feature'), fast_feature.to('cpu').numpy())

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='NTIRE')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--videos_dir', type=str, default='track3_data/DATA/wfr/Projects/THQA_3D/test/', help='The path of video')
    parser.add_argument('--datainfo', type=str, default='track3_data/thqa_ntire_testlist.csv', help='The csv path of video')
    parser.add_argument('--feature_save_folder', type=str, default='./NTIRE_test_feature/', help='The path of output')
    config = parser.parse_args()

    main(config)
