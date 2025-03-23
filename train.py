# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import VideoDataset_images_with_motion_features
from utils import performance_fit
from utils import L1RankLoss
from model import UGC_BVQA_model
from torchvision import transforms
import time
import csv



def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # breakpoint()
    if config.model_name == 'UGC_BVQA_model':
        print('The current model is ' + config.model_name)
        model = UGC_BVQA_model.resnet50(pretrained=True)
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    # Loss
    if config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()
    # 
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))
    

    frames_dir = config.frames_dir
    feature_dir = config.feature_dir
    datainfo_train = config.datainfo_train
    is_train = config.is_train
    
    # dataset
    transformations_train = transforms.Compose([transforms.Resize(config.resize), transforms.RandomCrop(config.crop_size), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    trainset = VideoDataset_images_with_motion_features(frames_dir, feature_dir, datainfo_train, transformations_train,
                                     'NTIRE', config.crop_size, 'SlowFast', is_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=False, num_workers=config.num_workers)
    # breakpoint()


    datainfo_eval = config.datainfo_eval
    transformations_eval = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    evalset = VideoDataset_images_with_motion_features(frames_dir, feature_dir, datainfo_eval, transformations_eval,
                                        'NTIRE', config.crop_size, 'SlowFast', is_train)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    # begin
    best_eval_criterion = -1  # SROCC min
    best_traing_loss_criterion = 9999
    best_eval = []
    best_train = []
    print('Starting training:')
    old_save_name = None
    old_training_save_name = None
    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, feature_3D, mos, _) in enumerate(train_loader):
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            labels = mos.to(device).float()
            
            # breakpoint()
            outputs = model(video, feature_3D)
            # 梯度置0
            optimizer.zero_grad()
            # 计算loss
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            # 方向传播
            loss.backward()
            # 优化器更新
            optimizer.step()
        
    
            # print log
            if (i+1) % (config.print_samples//config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                    (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                        avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        # 一轮的loss
        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        # 学习率更新
        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(evalset)])
            y_output = np.zeros([len(evalset)])
            for i, (video, feature_3D, mos, _) in enumerate(eval_loader):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label[i] = mos.item()
                outputs = model(video, feature_3D)
                y_output[i] = outputs.item()
            # eval
            eval_PLCC, eval_SRCC, eval_KRCC, eval_RMSE = performance_fit(label, y_output)
            print('Epoch {} completed. The result on the eval databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                eval_SRCC, eval_KRCC, eval_PLCC, eval_RMSE))

            # judge is or not best, save
            # if avg_loss < best_traing_loss_criterion:
            #     print("Update best model using best_traing_loss_criterion in epoch {}".format(epoch + 1))
            #     best_traing_loss_criterion = avg_loss
            #     best_train = [avg_loss]
            #     if not os.path.exists(config.ckpt_path):
            #         os.makedirs(config.ckpt_path)
            #     if epoch > 0:
            #         if os.path.exists(old_training_save_name):
            #             os.remove(old_training_save_name)
            #     save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
            #         config.database + '_' + config.loss_type + '_NR_v'+ str(config.exp_version) \
            #             + '_epoch_lr_%f_%d_SRCC_%f_l1w_%f.pth' % (config.conv_base_lr, epoch + 1, eval_SRCC, criterion.plcc_w))
            #     torch.save(model.state_dict(), save_model_name)
            #     old_training_save_name = save_model_name


            if eval_SRCC + eval_PLCC > best_eval_criterion:
                print("Update best model using best_eval_criterion in epoch {}".format(epoch + 1))
                best_eval_criterion = eval_SRCC + eval_PLCC
                best_eval = [eval_SRCC, eval_KRCC, eval_PLCC, eval_RMSE]
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                    config.database + '_' + config.loss_type + '_training_NR_v'+ str(config.exp_version) \
                        + 'epoch_%d_lr_%f_SRCC_%f.pth' % (config.conv_base_lr, epoch + 1, eval_SRCC))
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name


    print('Training completed.')
    print('The best training result on the training loss: {:.4f}'.format( \
        best_train[0]))
    print('The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_eval[0], best_eval[1], best_eval[2], best_eval[3]))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default = 0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default = 1000)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=10)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')
    # path
    parser.add_argument('--frames_dir', type=str, default='NTIRE_dataset/train/', help='The path of extracted frames')
    parser.add_argument('--feature_dir', type=str, default='NTIRE_feature/', help='The path of extracted feature')
    parser.add_argument('--datainfo_train', type=str, default='train_split.csv', help='The csv path of train_dataset')
    parser.add_argument('--datainfo_eval', type=str, default='val_split.csv', help='The csv path of train_dataset')
    parser.add_argument('--is_train', type=str, default=True)
    config = parser.parse_args()

    main(config)
