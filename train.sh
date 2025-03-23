LR=0.00001
exp_version=0
frames_dir=path/to/extracted/frames
feature_dir=path/to/extracted/feature
datainfo_train=path/to/train/csv
datainfo_eval=path/to/val/csv
ckpt_path=path/to/save/ckpt

CUDA_VISIBLE_DEVICES=0 python -u train.py \
                            --database NTIRE \
                            --model_name UGC_BVQA_model \
                            --conv_base_lr $LR \
                            --epochs 20 \
                            --train_batch_size 8 \
                            --print_samples 100 \
                            --num_workers 8 \
                            --ckpt_path $ckpt_path \
                            --decay_ratio 0.9 \
                            --decay_interval 1 \
                            --exp_version $exp_version \
                            --loss_type L1RankLoss \
                            --resize 520 \
                            --crop_size 448 \
                            --frames_dir $frames_dir \
                            --feature_dir $feature_dir \
                            --datainfo_train $datainfo_train \
                            --datainfo_eval $datainfo_eval \
                            | tee logs/train_lr_${LR}_version_${exp_version}.log
