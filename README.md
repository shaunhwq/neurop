
# Neural Color Operators for Sequential Image Retouching (ECCV2022) 

### Installation

```
conda create -n pt112c102 python=3.8
conda activate pt112c102
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip3 install opencv-python tqdm
```

### Running:
```
cd codes_pytorch
```

```
usage: demo_neurop.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR [--device DEVICE] --weights WEIGHTS

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to input folder containing images
  --output_dir OUTPUT_DIR
                        Path to output folder
  --device DEVICE       Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'
  --weights WEIGHTS     Path to weights
```

*[Yili Wang](https://yili.host), Xin Li, [Kun Xu](https://cg.cs.tsinghua.edu.cn/people/~kun/), Dongliang He, Qi Zhang, Fu Li, Errui Ding*


[[`arXiv`](https://arxiv.org/abs/2207.08080)] [[`project`](https://amberwangyili.github.io/neurop)] [[`doi`](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4401_ECCV_2022_paper.php)]

[[`Paddle Implementation`](codes_paddle)](**Offical**)

[[`Pytorch Implementation`](codes_pytorch)] 

[[`Jittor Implementation`](codes_jittor)]

<p align="center"> 
  <img src="figures/advantage.png">
</p><b>Left</b>: Compared with previous state-of-the-art methods, NeurOp achieves superior performance with only 28k parameters (~75% of CSRNet). <b>Right</b>: Strength Controllability Results. Our method can directly change the retouching output with intuitive control (i.e. directly modify the scalar strengths)



<p align="center"> 
	<img src="figures/result.png">
</p>


## Datasets

Pretrain data to initialize our neurOps is hosted on [百度网盘](https://pan.baidu.com/s/1r9zyYzD2-GuNGgu2dSKAYg) (code:pld9). 

### MIT-Adobe FiveK & PPR10K

 
We host all these data in [百度网盘](https://pan.baidu.com/s/1GD1VzZhSoRG6qOQ55u2buQ) (code:jvvq)

* There are two preprocessed versions of MIT-Adobe FiveK, in our paper, we refer them as MIT-Adobe FiveK-Dark (originally provided by [CSRNet](https://github.com/hejingwenhejingwen/CSRNet)) and MIT-Adobe FiveK-Lite (originally provided by [Distort-and-Recover](https://github.com/Jongchan/DISTORT-AND-RECOVER-CVPR18)). 

* The official PPR10K dataset link is [here](https://github.com/csjliang/PPR10K).


## Get Started

- Clone this repo

  ```
  git clone https://github.com/amberwangyili/neurop
  ```
  
- Download the Dataset from [百度网盘](https://pan.baidu.com/s/1GD1VzZhSoRG6qOQ55u2buQ) (code:jvvq) and unzip in project folder

  ```bash
  tree -L 2 neurop/datasets
  # the output should be like the following:
  datasets/
  ├── dataset-dark
  │   ├── testA
  │   ├── testB
  │   ├── trainA
  │   └── trainB
  ├── dataset-init
  │   ├── BC
  │   ├── EX
  │   └── VB
  ├── dataset-lite
  │   ├── testA
  │   ├── testB
  │   ├── trainA
  │   └── trainB
  └── dataset-ppr
      ├── ppr-a
      ├── ppr-b
      ├── ppr-c
      ├── testA
      ├── testM
      ├── trainA
      └── trainM
  ```

- Install Dependencies

  ```bash 
  cd neurop
  pip install -r requirements.txt 
  ```

## Test

1. We provide pretrained model weights for MIT-Adobe FiveK and PPR10K in `pretrain_models`

2. Run command:

   ```bash
   python test.py -config ./configs/test/<configuaration-name>.yaml 
   ```

3. The evaluation results will be in the `neurop/results` folder

## Train

1. Initialization individual neural color operators:
   
   ```bash
   python train.py -config ./configs/init_neurop.yaml 
   ```

2. Finetune with strength predictors:

   ```bash
   python train.py -config ./configs/train/<configuration-name>.yaml 
   ```



## BibTex

If you find neurOp useful in your research, please use the following BibTeX entry.

```BibTeX
    @inproceedings{wang2022neurop,
    author = {Wang, Yili and Li, Xin and Xu, Kun and He, Dongliang and Zhang, Qi and Li, Fu and Ding, Errui},
    title = {Neural Color Operators for Sequential Image Retouching},
    year = {2022},
    isbn = {978-3-031-19800-7},
    publisher = {Springer-Cham},
    url = {https://doi.org/10.1007/978-3-031-19800-7_3},
    doi = {10.1007/978-3-031-19800-7_3},
    booktitle = {Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XIX},
    numpages = {14},
    }
```

## Acknowledgement

NeurOp is licensed under a [MIT License](LICENSE).
