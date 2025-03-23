CUDA_VISIBLE_DEVICES=0 python -u extract_SlowFast_features_NTIRE.py \
                                --database NTIRE \
                                --model_name SlowFast \
                                --resize 224 \
                                --videos_dir /path/to/videos \
                                --datainfo /path/to/csv \
                                --feature_save_folder /path/to/output \
