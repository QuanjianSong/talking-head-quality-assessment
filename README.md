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
• <strong>2024.05</strong>: 🔥 The official code of our team -- 'FocusQ',is now available.
• <strong>2024.04</strong>: 🔥 Our team -- 'FocusQ', achieved 6th place in the <a href="https://codalab.lisn.upsaclay.fr/competitions/21555" target="_blank">2025 XGC Quality Assessment - Track 3: Talking Head</a>.
</pre>


  
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
#### ► 1.Extract video frames from the training dataset.
You can manually modify the relevant file paths in the `extract_frames.sh` file, and then execute the following command to extract video frames from the training dataset:
```
bash extract_frames.sh # for training dataset
```

#### ► 2.Extract motion features from the training dataset
You can manually modify the relevant file paths in the `extract_features.sh` file, and then execute the following command to extract motion features from the training dataset:
```
bash extract_features.sh # for training dataset
```

#### ► 3.Train the model
You can manually modify the relevant file paths in the `train.sh` file, and then execute the following command to start training:
```
bash train.sh
```
Please note that our code runs on a single H100 with 80GB of VRAM. If your VRAM is lower, please reduce the batch_size and adjust the learning rate accordingly.

## 🌈 Test
### ►3.Extract video frames from the test dataset.
Before testing, you also need to modify the relevant paths in the `extract_frames.sh` file, and then execute the following command to extract video frames from the test dataset:
```
bash extract_frames.sh # for test dataset
```

### ►3.Extract motion features from the test dataset
Before testing, you also need to modify the relevant paths in the `extract_features.sh` file, and then execute the following command to extract video frames from the test dataset:
```
bash extract_features.sh # for test dataset
```

### ►3.Start testing
Finally, you can modify the relevant paths in `test.sh`, and then run the following command to start the testing:
```
bash test.sh
```

## 🎓 Bibtex
If you find this code helpful for your research, please cite:
```
@misc{liu2025ntire2025xgcquality,
      title={NTIRE 2025 XGC Quality Assessment Challenge: Methods and Results}, 
      author={Xiaohong Liu and Xiongkuo Min and Qiang Hu and Xiaoyun Zhang and Jie Guo and Guangtao Zhai and Shushi Wang and Yingjie Zhou and Lu Liu and Jingxin Li and Liu Yang and Farong Wen and Li Xu and Yanwei Jiang and Xilei Zhu and Chunyi Li and Zicheng Zhang and Huiyu Duan and Xiele Wu and Yixuan Gao and Yuqin Cao and Jun Jia and Wei Sun and Jiezhang Cao and Radu Timofte and Baojun Li and Jiamian Huang and Dan Luo and Tao Liu and Weixia Zhang and Bingkun Zheng and Junlin Chen and Ruikai Zhou and Meiya Chen and Yu Wang and Hao Jiang and Xiantao Li and Yuxiang Jiang and Jun Tang and Yimeng Zhao and Bo Hu and Zelu Qi and Chaoyang Zhang and Fei Zhao and Ping Shi and Lingzhi Fu and Heng Cong and Shuai He and Rongyu Zhang and Jiarong He and Zongyao Hu and Wei Luo and Zihao Yu and Fengbin Guan and Yiting Lu and Xin Li and Zhibo Chen and Mengjing Su and Yi Wang and Tuo Chen and Chunxiao Li and Shuaiyu Zhao and Jiaxin Wen and Chuyi Lin and Sitong Liu and Ningxin Chu and Jing Wan and Yu Zhou and Baoying Chen and Jishen Zeng and Jiarui Liu and Xianjin Liu and Xin Chen and Lanzhi Zhou and Hangyu Li and You Han and Bibo Xiang and Zhenjie Liu and Jianzhang Lu and Jialin Gui and Renjie Lu and Shangfei Wang and Donghao Zhou and Jingyu Lin and Quanjian Song and Jiancheng Huang and Yufeng Yang and Changwei Wang and Shupeng Zhong and Yang Yang and Lihuo He and Jia Liu and Yuting Xing and Tida Fang and Yuchun Jin},
      year={2025},
      eprint={2506.02875},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.02875}, 
}
```
