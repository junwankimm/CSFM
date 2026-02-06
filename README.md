# Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching (CSFM)


[📄 Paper](https://arxiv.org/abs/2602.05951) · [🌐 Project Page](https://junwankimm.github.io/CSFM) · [🤗 Dataset](https://huggingface.co/datasets/junwann/CSFM-ImageNet1K-Caption)


<!-- --- -->

<!-- ## 📌 Overview -->

This repository provides the official PyTorch implementation of **Condition-Dependent Source Flow Matching (CSFM)**. 
<p align="center">
  <img src="assets/csfm.png" alt="CSFM Overview" width="500">
</p>

---

### 📨 News
🚀 05/Feb/26 - Released training and evaluation code of CSFM!

### 🛠 Environment

```bash
conda create -n csfm python=3.10 -y
conda activate csfm
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 
pip install -r requirements.txt
```

### 📦 Dataset: CSFM-ImageNet1K-Caption
Image data: [ImageNet-1K dataset](https://www.image-net.org/).

Text Caption data: [CSFM-ImageNet1K-Caption](https://huggingface.co/datasets/junwann/CSFM-ImageNet1K-Caption)
    
  ```bash
  hf download junwann/CSFM-ImageNet1K-Caption \
    --repo-type=dataset \
    --local-dir captions
  ```
**Important**: Make sure that the image paths in the caption files match your local ImageNet data paths.

<!-- SDVAE
```bash
``` -->

### ⚙️ Setup: RAE decoders
We used RAE decoders as default

```bash
hf download nyu-visionx/RAE-collections \
  --local-dir models 
```

**Important**: Make sure the model and stats paths match those defined in the config file.

### 🔥 Training  

`src/train_t2i.py` is compatible with both GPU and TPU(TorchXLA).

enable `--use_xla` for TPU training. We tested on torch-xla==2.8.1

Configs
- CSFM: `configs/csfm.yaml`     
- Standard FM: `configs/fm.yaml`   
  
```bash
export ENTITY="your_wandb_entity"
export PROJECT="your_wandb_project"
export WANDB_KEY="your_wandb_key"
exp_name=""
imagenet1k_train_path=""
torchrun --standalone --nnodes=1 --nproc_per_node=4 src/train_t2i.py \
    --config configs/csfm.yaml \
    --data_path $imagenet1k_train_path \
    --exp_name $exp_name \
    --jsonl_path captions/train.jsonl \
    --jsonl_path_val val_prompt.json \
    --exps exps \
    --last_ckpt_every 5000 \
    --precision bf16 \
    --wandb;
```
**Experiment Directory Structure**  
```bash
exps_dir/
├─ exp1/
│  ├─ checkpoints/
│  │  ├─ 010000.pt
│  │  └─ last-015000.pt
│  └─ config.yaml
├─ exp2/
│  ├─ checkpoints/
│  │  └─ ...
│  └─ config.yaml
└─ ...
```


### 🧪 Evaluation

Download ImageNet statistics (256×256 shown here) for FID

```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```
Sample/Eval Code
```bash
exps_dir=''
exp_name=''
sample_dir=''
exp_name=''
train_step=100000
sampling_step=50
batch_per_gpu=250
fid_stat_npz=''
csv_path='metrics.csv'
torchrun --standalone --nnodes=1 --nproc_per_node=4 src/test_t2i.py \
    --csv_path $csv_path \
    --exps_dir $exps_dir \
    --sample_dir $sample_dir \
    --exp_name $exp_name \
    --train_step $train_step \
    --num_samples 50000 \
    --sampling_method euler \
    --sampling_num_steps $sampling_step \
    --per_proc_batch_size $batch_per_gpu \
    --fid_stat_npz $fid_stat_npz \
    --jsonl_path captions/val.jsonl \
    --precision fp32 \
    --tf32 \
    --global_seed 0 \

```

To follow the ADM evaluation use `--make_npz` to create npz file and follow this setup

```bash
git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion/evaluation
conda create -n adm-fid python=3.10
conda activate adm-fid
pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
```

### Acknowledgement

This code is built upon the following repositories:
- [RAE](https://github.com/bytetriper/RAE): for RAE and diffusion implementation
- [Lumina-Image 2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0): for Unified Next-DiT implementation

The toy experiments in the paper are conducted upon following repository:
- [C2OT](https://github.com/hkchengrex/C2OT): for toy experiments (not provided in this repo)


### BibTeX
```bash
```
