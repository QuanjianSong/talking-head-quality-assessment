<div align="center">
<h1>
🚀 XGC Quality Assessment - Track 3: Talking Head (NTIRE Workshop and Challenges @ CVPR 2025) [Official Code of PyTorch]
</h1>
    


<p align="center">
    <span>
        <a href="https://arxiv.org/abs/2506.02875" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202506.02875-NTIRE_2025-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span>
        <a href="https://github.com/zyj-2000/THQA-NTIRE" target="_blank"> 
        <img src='https://img.shields.io/badge/Project_Page-NTIRE_2025-green' alt='Project Page'></a> &emsp;  &emsp; 
    </span>
    <span>
        <a href="https://huggingface.co/papers/2506.02875" target="_blank"> 
        <img src='https://img.shields.io/badge/Hugging_Face-NTIRE_2025-yellow' alt='Hugging Face'></a> &emsp;  &emsp; 
    </span>

  
</p>

</div>

---

## 🎉 News
<pre>
• <strong>2024.06</strong>: 🔥 Our report has been accepted to the 2025 CVPR Workshop.   
• <strong>2024.05</strong>: 🔥 The official code of our team -- 'FocusQ',is now available.
• <strong>2024.04</strong>: 🔥 Our team -- 'FocusQ', achieved 6th place in the <a href="https://codalab.lisn.upsaclay.fr/competitions/21555" target="_blank">2025 XGC Quality Assessment - Track 3: Talking Head</a>.
</pre>

## 🎬 Overview
Official Implementation Code of Our Team 'FocusQ' in XGC Quality Assessment - Track 3: Talking Head.
The detailed information can be found in the <a href="https://github.com/zyj-2000/THQA-NTIRE" target="_blank">official repository</a>.
  
## 🔧 Environment
```
# Git clone the repo
git clone https://github.com/QuanjianSong/talking-head-quality-assessment.git

# Installation with the requirement.txt
conda create -n NTIRE25 python=3.10
conda activate NTIRE25
pip install -r requirements.txt

# Or installation with the environment.yaml
conda env create -f environment.yaml
```

## 📖 Dataset
You can download the XGC Quality Assessment - Track 3: Talking Head dataset from this <a href="https://huggingface.co/datasets/zyj2000/THQA-NTIRE/tree/main" target="_blank">link</a>.




## 🔥 Train
#### • 1.Extract video frames from the training dataset.
You can manually modify the relevant file paths in the `extract_frames.sh` file, and then execute the following command to extract video frames from the training dataset:
```
bash extract_frames.sh # for training dataset
```

#### • 2.Extract motion features from the training dataset
You can manually modify the relevant file paths in the `extract_features.sh` file, and then execute the following command to extract motion features from the training dataset:
```
bash extract_features.sh # for training dataset
```

#### • 3.Train the model
You can manually modify the relevant file paths in the `train.sh` file, and then execute the following command to start training:
```
bash train.sh
```
Please note that our code runs on a single H100 with 80GB of VRAM. If your VRAM is lower, please reduce the batch_size and adjust the learning rate accordingly.

## 🚀 Test
#### • 1.Extract video frames from the test dataset.
Before testing, you also need to modify the relevant paths in the `extract_frames.sh` file, and then execute the following command to extract video frames from the test dataset:
```
bash extract_frames.sh # for test dataset
```

#### • 2.Extract motion features from the test dataset
Before testing, you also need to modify the relevant paths in the `extract_features.sh` file, and then execute the following command to extract video frames from the test dataset:
```
bash extract_features.sh # for test dataset
```

#### • 3.Start testing
Finally, you can modify the relevant paths in `test.sh`, and then run the following command to start the testing:
```
bash test.sh
```

## 🎓 Bibtex
🤗 If you find this code helpful for your research, please cite:
```
@inproceedings{liu2025ntire,
  title={NTIRE 2025 XGC Quality Assessment Challenge: Methods and Results},
  author={Liu, Xiaohong and Min, Xiongkuo and Hu, Qiang and Zhang, Xiaoyun and Guo, Jie and Zhai, Guangtao and Wang, Shushi and Zhou, Yingjie and Liu, Lu and Li, Jingxin and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1389--1402},
  year={2025}
}
```
