<div align="center">
<h1>
XGC Quality Assessment - Track 3: Talking Head (NTIRE Workshop and Challenges @ CVPR 2025) [Official Code of PyTorch]
</h1>
# [🚀XGC Quality Assessment - Track 3: Talking Head (NTIRE Workshop and Challenges @ CVPR 2025)](https://github.com/zyj-2000/THQA-NTIRE)

<p align="center">
    <span>
        <a href="https://huggingface.co/papers/2506.02875" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202506.02875-NTIRE_2025-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span>
        <a href="https://github.com/zyj-2000/THQA-NTIRE" target="_blank"> 
        <img src='XXX' alt='Project Page'></a> &emsp;  &emsp; 
    </span>

    <span>
        <a href="https://huggingface.co/papers/2506.02875" target="_blank"> 
        <img src='Hugging_Face-NTIRE_2025-yellow' alt='Hugging Face'></a> &emsp;  &emsp; 
    </span>

  
</p>

</div>

---

## 🔥🔥🔥 News
<pre>
• <strong>2024.10.26</strong>: 🔥 Our team, 'FocusQ' achieved 6th place in the <a href="[https://example.com](https://github.com/zyj-2000/THQA-NTIRE)">2025 XGC Quality Assessment - Track 3: Talking Head.</a>
• <strong>2024.10.26</strong>: 🔥 The official code of our team is now available.
</pre>

[2025 XGC Quality Assessment - Track 3: Talking Head](https://github.com/zyj-2000/THQA-NTIRE).
  
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
Please note that our code runs on a single H100 with 80GB of VRAM. If your VRAM is lower, please reduce the batch_size and adjust the learning rate accordingly.

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
