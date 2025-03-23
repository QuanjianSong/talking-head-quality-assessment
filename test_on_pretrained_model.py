import argparse
import os
import numpy as np
import torch
import torch.nn
from torchvision import transforms
from model import UGC_BVQA_model
from utils import performance_fit
from data_loader import VideoDataset_images_with_motion_features
from fvcore.nn import FlopCountAnalysis
import random

# fix seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.model_name == 'UGC_BVQA_model':
        print('The current model is ' + config.model_name)
        model = UGC_BVQA_model.resnet50(pretrained=False)
    model = model.to(device)
    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))

    datainfo_test = config.datainfo_test
    frames_dir = config.frames_dir
    feature_dir = config.feature_dir
    is_train = config.is_train

    # dataset
    transformations_test = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),\
        transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    testset = VideoDataset_images_with_motion_features(frames_dir, feature_dir, datainfo_test, \
        transformations_test, config.database, 448, config.feature_type, is_train)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    # # 计算模型的参数数量
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"模型的总参数数量：{total_params}")

    # test
    with torch.no_grad():
        model.eval()
        label = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        videos_name = []
        # for loop
        for i, (video, feature_3D, mos, video_name) in enumerate(test_loader):
            print(video_name[0])
            videos_name.append(video_name)
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            label[i] = mos.item()
            # flops = FlopCountAnalysis(model, (video, feature_3D))
            # print(f"flops: {flops.total()}")

            outputs = model(video, feature_3D)
            y_output[i] = outputs.item()

        # save to .txt
        with open(config.output, "w", encoding="utf-8") as file:
            for video_name, output in zip(videos_name, y_output):
                item = f'{video_name[0]},{output}'
                file.write(item + "\n")  # 确保所有数据转换为字符串
        print(f"数据已成功写入 {config.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str, default='NTIRE')
    parser.add_argument('--model_name', type=str, default='UGC_BVQA_model')
    parser.add_argument('--num_workers', type=int, default=6)
    # misc
    parser.add_argument('--feature_type', type=str, default='SlowFast')
    parser.add_argument('--gpu_ids', type=list, default=0)
    # path
    parser.add_argument('--trained_model', type=str, default='you_model_path', help='The path of your models')
    parser.add_argument('--frames_dir', type=str, default='test/', help='The path of extracted frames')
    parser.add_argument('--feature_dir', type=str, default='NTIRE_test_feature/', help='The path of extracted feature')
    parser.add_argument('--datainfo_test', type=str, default='thqa_ntire_testlist.csv')
    parser.add_argument('--output', type=str, default='output.txt', help='The path of output')
    parser.add_argument('--is_train', type=str, default=False)
    config = parser.parse_args()

    main(config)
