## Title
My implementation of the demonstration for the paper: KDD2020 Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion.
No updates for refactoring in future, as I'll try to rewrite all using the new CRSLab framework and implement other papers.

## Usage
First please ensure to download the trained parameters for the recommendation module and the conversation module, as this repo only serves as a live-time user-interaction demonstration stystem.
Please download from: 

https://drive.google.com/drive/folders/1dNkeAGzprnaVfZegxeLrtp9bxEYCrUY9?usp=sharing

To use the command line demo system, please run on linux:
```
bash command_demo.sh
```

To use the website UI demo system, please run on linux:
```
bash webUI_demo.sh
```

Please insure you have GPU and cuda to test the model.

## Reference
@inproceedings{DBLP:conf/kdd/ZhouZBZWY20,
  author    = {Kun Zhou and
               Wayne Xin Zhao and
               Shuqing Bian and
               Yuanhang Zhou and
               Ji{-}Rong Wen and
               Jingsong Yu},
  title     = {Improving Conversational Recommender Systems via Knowledge Graph based
               Semantic Fusion},
  booktitle = {{KDD} '20: The 26th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Virtual Event, CA, USA, August 23-27, 2020},
  pages     = {1006--1014},
  year      = {2020},
  url       = {https://dl.acm.org/doi/10.1145/3394486.3403143}
}