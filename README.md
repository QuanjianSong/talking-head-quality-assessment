# [🚀XGC Quality Assessment - Track 3: Talking Head (NTIRE Workshop and Challenges @ CVPR 2025)](https://github.com/zyj-2000/THQA-NTIRE)
This is the official implementation code from our team, 'FocusQ', which ranked 6th.
## 1. Prepare environment
### Installation with the requirement.txt
```
conda create -n NTIRE25 python=3.10
conda activate NTIRE25
pip install -r requirements.txt
```

### Installation with the environment.yaml
```
conda env create -f environment.yaml
```

## 2. Prepare training
### 2.1 Extract video frames from the training dataset.
You can manually modify the relevant file paths in the `extract_frames.sh` file, and then execute the following command to extract video frames from the training dataset:
```
bash extract_frames.sh # for training dataset
```

### 2.2 Extract motion features from the training dataset
You can manually modify the relevant file paths in the `extract_features.sh` file, and then execute the following command to extract motion features from the training dataset:
```
bash extract_features.sh # for training dataset
```

### 2.3 Train the model
You can manually modify the relevant file paths in the `train.sh` file, and then execute the following command to start training:
```
bash train.sh
```

## 3. Test the model
### 3.1 Extract video frames from the test dataset.
Before testing, you also need to modify the relevant paths in the `extract_frames.sh` file, and then execute the following command to extract video frames from the test dataset:
```
bash extract_frames.sh # for test dataset
```

### 3.2 Extract motion features from the test dataset
Before testing, you also need to modify the relevant paths in the `extract_features.sh` file, and then execute the following command to extract video frames from the test dataset:
```
bash extract_features.sh # for test dataset
```

### 3.3 Start testing
Finally, you can modify the relevant paths in `test.sh`, and then run the following command to start the testing:
```
bash test.sh
```
