# Deformable DETR

By [Xizhou Zhu](https://scholar.google.com/citations?user=02RXI00AAAAJ),  [Weijie Su](https://www.weijiesu.com/),  [Lewei Lu](https://www.linkedin.com/in/lewei-lu-94015977/), [Bin Li](http://staff.ustc.edu.cn/~binli/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Jifeng Dai](https://jifengdai.org/).

This repository is an official implementation of the paper [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).


## Introduction

**TL; DR.** Deformable DETR is an efficient and fast-converging end-to-end object detector. It mitigates the high complexity and slow convergence issues of DETR via a novel sampling-based efficient attention mechanism.  

![deformable_detr](./figs/illustration.png)

![deformable_detr](./figs/convergence.png)

**Abstract.** DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10× less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach.

## License

This project is released under the [Apache 2.0 license](./LICENSE).

## Changelog

See [changelog.md](./docs/changelog.md) for detailed logs of major changes. 


## Citing Deformable DETR
If you find Deformable DETR useful in your research, please consider citing:
```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```

## Main Results

| <sub><sub>Method</sub></sub>   | <sub><sub>Epochs</sub></sub> | <sub><sub>AP</sub></sub> | <sub><sub>AP<sub>S</sub></sub></sub> | <sub><sub>AP<sub>M</sub></sub></sub> | <sub><sub>AP<sub>L</sub></sub></sub> | <sub><sub>params<br>(M)</sub></sub> | <sub><sub>FLOPs<br>(G)</sub></sub> | <sub><sub>Total<br>Train<br>Time<br>(GPU<br/>hours)</sub></sub> | <sub><sub>Train<br/>Speed<br>(GPU<br/>hours<br/>/epoch)</sub></sub> | <sub><sub>Infer<br/>Speed<br/>(FPS)</sub></sub> | <sub><sub>Batch<br/>Infer<br/>Speed<br>(FPS)</sub></sub> | <sub><sub>URL</sub></sub>                     |
| ----------------------------------- | :----: | :--: | :----: | :---: | :------------------------------: | :--------------------:| :----------------------------------------------------------: | :--: | :---: | :---: | ----- | ----- |
| <sub><sub>Faster R-CNN + FPN</sub></sub> | <sub>109</sub> | <sub>42.0</sub> | <sub>26.6</sub> | <sub>45.4</sub> | <sub>53.4</sub> | <sub>42</sub> | <sub>180</sub> | <sub>380</sub> | <sub>3.5</sub> | <sub>25.6</sub> | <sub>28.0</sub> | <sub>-</sub> |
| <sub><sub>DETR</sub></sub> | <sub>500</sub> | <sub>42.0</sub> | <sub>20.5</sub> | <sub>45.8</sub> | <sub>61.1</sub> | <sub>41</sub> | <sub>86</sub> | <sub>2000</sub> | <sub>4.0</sub> |     <sub>27.0</sub>       |         <sub>38.3</sub>           | <sub>-</sub> |
| <sub><sub>DETR-DC5</sub></sub>      | <sub>500</sub> | <sub>43.3</sub> | <sub>22.5</sub> | <sub>47.3</sub> | <sub>61.1</sub> | <sub>41</sub> |<sub>187</sub>|<sub>7000</sub>|<sub>14.0</sub>|<sub>11.4</sub>|<sub>12.4</sub>| <sub>-</sub> |
| <sub><sub>DETR-DC5</sub></sub>      | <sub>50</sub> | <sub>35.3</sub> | <sub>15.2</sub> | <sub>37.5</sub> | <sub>53.6</sub> | <sub>41</sub> |<sub>187</sub>|<sub>700</sub>|<sub>14.0</sub>|<sub>11.4</sub>|<sub>12.4</sub>| <sub>-</sub> |
| <sub><sub>DETR-DC5+</sub></sub>     | <sub>50</sub> | <sub>36.2</sub> | <sub>16.3</sub> | <sub>39.2</sub> | <sub>53.9</sub> | <sub>41</sub> |<sub>187</sub>|<sub>700</sub>|<sub>14.0</sub>|<sub>11.4</sub>|<sub>12.4</sub>| <sub>-</sub> |
| **<sub><sub>Deformable DETR<br>(single scale)</sub></sub>** | <sub>50</sub> | <sub>39.4</sub> | <sub>20.6</sub> | <sub>43.0</sub> | <sub>55.5</sub> | <sub>34</sub> |<sub>78</sub>|<sub>160</sub>|<sub>3.2</sub>|<sub>27.0</sub>|<sub>42.4</sub>| <sub>[config](./configs/r50_deformable_detr_single_scale.sh)<br/>[log](https://drive.google.com/file/d/1n3ZnZ-UAqmTUR4AZoM4qQntIDn6qCZx4/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU/view?usp=sharing)</sub> |
| **<sub><sub>Deformable DETR<br>(single scale, DC5)</sub></sub>** | <sub>50</sub> | <sub>41.5</sub> | <sub>24.1</sub> | <sub>45.3</sub> | <sub>56.0</sub> | <sub>34</sub> |<sub>128</sub>|<sub>215</sub>|<sub>4.3</sub>|<sub>22.1</sub>|<sub>29.4</sub>| <sub>[config](./configs/r50_deformable_detr_single_scale_dc5.sh)<br/>[log](https://drive.google.com/file/d/1-UfTp2q4GIkJjsaMRIkQxa5k5vn8_n-B/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk/view?usp=sharing)</sub> |
| **<sub><sub>Deformable DETR</sub></sub>** | <sub>50</sub> | <sub>44.5</sub> | <sub>27.1</sub> | <sub>47.6</sub> | <sub>59.6</sub> | <sub>40</sub> |<sub>173</sub>|<sub>325</sub>|<sub>6.5</sub>|<sub>15.0</sub>|<sub>19.4</sub>|<sub>[config](./configs/r50_deformable_detr.sh)<br/>[log](https://drive.google.com/file/d/18YSLshFjc_erOLfFC-hHu4MX4iyz1Dqr/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view?usp=sharing)</sub>                   |
| **<sub><sub>+ iterative bounding box refinement</sub></sub>** | <sub>50</sub> | <sub>46.2</sub> | <sub>28.3</sub> | <sub>49.2</sub> | <sub>61.5</sub> | <sub>41</sub> |<sub>173</sub>|<sub>325</sub>|<sub>6.5</sub>|<sub>15.0</sub>|<sub>19.4</sub>|<sub>[config](./configs/r50_deformable_detr_plus_iterative_bbox_refinement.sh)<br/>[log](https://drive.google.com/file/d/1DFNloITi1SFBWjYzvVEAI75ndwmGM1Uj/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view?usp=sharing)</sub> |
| **<sub><sub>++ two-stage Deformable DETR</sub></sub>** | <sub>50</sub> | <sub>46.9</sub> | <sub>29.6</sub> | <sub>50.1</sub> | <sub>61.6</sub> | <sub>41</sub> |<sub>173</sub>|<sub>340</sub>|<sub>6.8</sub>|<sub>14.5</sub>|<sub>18.8</sub>|<sub>[config](./configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh)<br/>[log](https://drive.google.com/file/d/1ozi0wbv5-Sc5TbWt1jAuXco72vEfEtbY/view?usp=sharing) <br/>[model](https://drive.google.com/file/d/15I03A7hNTpwuLNdfuEmW9_taZMNVssEp/view?usp=sharing)</sub> |

*Note:*

1. All models of Deformable DETR are trained with total batch size of 32. 
2. Training and inference speed are measured on NVIDIA Tesla V100 GPU.
3. "Deformable DETR (single scale)" means only using res5 feature map (of stride 32) as input feature maps for Deformable Transformer Encoder.
4. "DC5" means removing the stride in C5 stage of ResNet and add a dilation of 2 instead.
5. "DETR-DC5+" indicates DETR-DC5 with some modifications, including using Focal Loss for bounding box classification and increasing number of object queries to 300.
6. "Batch Infer Speed" refer to inference with batch size = 4  to maximize GPU utilization.
7. The original implementation is based on our internal codebase. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.


## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```

#### Training on multiple nodes

For example, the command for training Deformable DETR on 2 nodes of each with 8 GPUs is as following:

On node 1:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=0 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

On node 2:

```bash
MASTER_ADDR=<IP address of node 1> NODE_RANK=1 GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 16 ./configs/r50_deformable_detr.sh
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 8 configs/r50_deformable_detr.sh
```

Or 2 nodes of  each with 8 GPUs:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> deformable_detr 16 configs/r50_deformable_detr.sh
```
#### Some tips to speed-up training
* If your file system is slow to read images, you may consider enabling '--cache_mode' option to load whole dataset into memory at the beginning of training.
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 3' or '--batch_size 4'.

### Evaluation

You can get the config file and pretrained model of Deformable DETR (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 validation set:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

You can also run distributed evaluation by using ```./tools/run_dist_launch.sh``` or ```./tools/run_dist_slurm.sh```.

---

## Sketch-Query Detection Baseline

This section describes the minimal sketch-guided detection baseline built on top of Deformable DETR.

**Idea.** After the decoder produces 300 object-query features (`hs[-1]`, 256-d), a linear projection layer maps them into CLIP's 512-d embedding space. At inference, each proposal is scored by its cosine similarity with the CLIP visual embedding of a sketch query. No category labels are used at test time.

```
Image ──▶ Def-DETR backbone + decoder ──▶ query features [B, 300, 256]
                                                    │ query_clip_proj (256→512)
                                                    ▼
                                          query embeddings [B, 300, 512]
                                                    │ cosine similarity
Sketch ──▶ CLIP ViT-B/32 ───────────────▶ sketch embedding [B, 512]
                                                    │
                                          ranked proposals → mAP / Recall@k
```

### Environment setup

All scripts source `.env` from the project root for the following variables:

```bash
# .env  (already present — edit paths if needed)
export COCO_HOME="/mnt/1tb/data/coco"
export SKETCH_HOME="/mnt/1tb/data/quickdraw/sketchrnn"   # QuickDraw .npy files
export PROJECT_HOME="/home/rahul/arka/clip_ddetr_base"
```

Activate the conda environment before running anything:

```bash
conda activate clip_ddetr
```

### Checkpoints layout

```
checkpoints/
├── clip_model/
│   ├── ViT-B-32.pt               # CLIP visual encoder
│   └── text_embeddings.pkl       # pre-computed CLIP text embeddings
├── r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
└── r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth
```

### World-split variants

| Mode | Description | When to use |
|---|---|---|
| `open` **(default)** | Train on **seen** categories only; eval categories are held out | Rigorous baseline — tests generalisation to unseen sketch queries |
| `closed` | Train on **all** categories, including eval ones | Upper-bound reference; numbers will be inflated for held-out categories |

Both scripts accept `WORLD` as the third positional argument.

### Step 1 — Train the projection layer

Freezes all def-DETR weights and trains only `query_clip_proj` (256 → 512, ~131 K params) using pre-computed CLIP text embeddings as alignment targets.

Each script carries its own `#SBATCH` headers so it runs identically whether invoked locally or submitted to Slurm:

```bash
bash  scripts/train_proj.sh [RESUME] [SKETCH_DS] [WORLD]   # local GPU
sbatch scripts/train_proj.sh                                # Slurm (uses #SBATCH defaults)
```

| Positional arg | Default | Options |
|---|---|---|
| `RESUME` | `checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth` | any def-DETR `.pth` |
| `SKETCH_DS` | `qd` | `qd` (QuickDraw), `sk` (Sketchy) |
| `WORLD` | `open` | `open`, `closed` |

**Examples**

```bash
# Open-world QuickDraw — local GPU
bash scripts/train_proj.sh

# Same, submitted to Slurm
sbatch scripts/train_proj.sh

# Closed-world upper-bound
bash scripts/train_proj.sh checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth qd closed

# Sketchy, open-world
bash scripts/train_proj.sh checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth sk open

# Pass extra Python flags after the three positional args
bash scripts/train_proj.sh checkpoints/....pth qd open --epochs 20 --lr 5e-4
```

Checkpoints are saved to `outputs/clip_proj_aligned_{WORLD}_{SKETCH_DS}/` after every epoch:

```
outputs/clip_proj_aligned_open_qd/
├── checkpoint.pth        # latest epoch (overwritten each epoch)
├── checkpoint0000.pth
├── checkpoint0001.pth
└── ...
```

Key Python flags (pass after the three positional args):

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch_size` | `4` | Batch size |
| `--debug_size` | `0` | Truncate dataset to N samples (0 = full) |

---

### Step 2 — Evaluate the baseline

Ranks each image's 300 proposals by cosine similarity with the sketch CLIP embedding and reports:

- **Recall@1 / @5 / @10** at IoU ≥ 0.5
- **COCO-style mAP**, AP@50, APS, APM, APL (class-agnostic, cosine-sim as confidence score)

```bash
bash  scripts/eval_baseline.sh [RESUME] [SKETCH_DS] [WORLD]   # local GPU
sbatch scripts/eval_baseline.sh                                # Slurm
```

| Positional arg | Default | Options |
|---|---|---|
| `RESUME` | `outputs/clip_proj_aligned_{WORLD}_{SKETCH_DS}/checkpoint.pth` | any trained `.pth` |
| `SKETCH_DS` | `qd` | `qd` (QuickDraw), `sk` (Sketchy) |
| `WORLD` | `open` | `open`, `closed` |

**Examples**

```bash
# Open-world QuickDraw — local GPU (matches default training run)
bash scripts/eval_baseline.sh

# Same, submitted to Slurm
sbatch scripts/eval_baseline.sh

# Evaluate the closed-world checkpoint
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_closed_qd/checkpoint.pth qd closed

# Specific epoch checkpoint
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_qd/checkpoint0004.pth qd open

# Sketchy, open-world
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_sk/checkpoint.pth sk open

# Change IoU threshold or top-k values
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_qd/checkpoint.pth qd open \
    --iou_thresh 0.25 --topk 1 3 5 10
```

**Expected output format**

```
============================================================
Sketch-Query Detection Baseline  |  N samples
============================================================
  Recall@ 1 (IoU≥0.5): 0.XXXX  (X/N)
  Recall@ 5 (IoU≥0.5): 0.XXXX  (X/N)
  Recall@10 (IoU≥0.5): 0.XXXX  (X/N)

============================================================
COCO-style class-agnostic mAP (cosine-sim as confidence score):
============================================================
 Average Precision  (AP) @[ IoU=0.50:0.95 | ... ]
 ...
  mAP    : 0.XXXX
  AP@50  : 0.XXXX
  AP@75  : 0.XXXX
  APS    : 0.XXXX
  APM    : 0.XXXX
  APL    : 0.XXXX
```

Key flags (pass after positional args to override defaults):

| Flag | Default | Description |
|---|---|---|
| `--topk` | `1 5 10` | Space-separated list of k values for Recall@k |
| `--iou_thresh` | `0.5` | IoU threshold for Recall@k |
| `--batch_size` | `1` | Inference batch size |
| `--clip_checkpoint` | `checkpoints/clip_model/ViT-B-32.pt` | CLIP model path |
| `--output_dir` | `outputs/eval_baseline` | Output directory |

---

### Quick end-to-end run

```bash
# Rigorous baseline (open-world, QuickDraw) — local GPU
bash scripts/train_proj.sh     # → outputs/clip_proj_aligned_open_qd/
bash scripts/eval_baseline.sh

# Same, on Slurm (no changes needed — #SBATCH headers are built into the scripts)
sbatch scripts/train_proj.sh
sbatch scripts/eval_baseline.sh

# Upper-bound comparison (closed-world, same dataset)
bash  scripts/train_proj.sh checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth qd closed
bash  scripts/eval_baseline.sh outputs/clip_proj_aligned_closed_qd/checkpoint.pth qd closed
```
