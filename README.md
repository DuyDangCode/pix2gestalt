# pix2gestalt: Amodal Segmentation by Synthesizing Wholes
### CVPR 2024 (Highlight)
### [Project Page](https://gestalt.cs.columbia.edu/)  | [Paper](https://arxiv.org/pdf/2401.14398.pdf) | [arXiv](https://arxiv.org/abs/2401.14398) | [Weights](https://huggingface.co/cvlab/pix2gestalt-weights) | [Citation](https://github.com/cvlab-columbia/pix2gestalt#citation)

[pix2gestalt: Amodal Segmentation by Synthesizing Wholes](https://gestalt.cs.columbia.edu/)  
 [Ege Ozguroglu](https://egeozguroglu.github.io/)<sup>1</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/)<sup>1</sup>, [Dídac Surís](https://www.didacsuris.com/)<sup>1</sup>, [Dian Chen](https://scholar.google.com/citations?user=zdAyna8AAAAJ&hl=en)<sup>2</sup>, [Achal Dave](https://www.achaldave.com/)<sup>2</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup> <br>
 <sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute

![teaser](./assets/teaser.gif "Teaser")

## Updates
- **[New]** Added automated evaluation benchmark pipeline for **BSDS-A** (`run_eval_bsds.sh` & `eval_bsds.py`) with support for PyTorch 2.x, CUDA 12.1, and modern RTX 40-series (`sm_89`) GPUs.
- We have released our [training script](https://github.com/cvlab-columbia/pix2gestalt?tab=readme-ov-file#training), [dataset](https://github.com/cvlab-columbia/pix2gestalt?tab=readme-ov-file#dataset), and [Gradio demo](https://github.com/cvlab-columbia/pix2gestalt?tab=readme-ov-file#inference-and-weights) with inference instructions.
- Pretrained models are released on [Huggingface](https://huggingface.co/cvlab/pix2gestalt-weights), more details provided [here](https://github.com/cvlab-columbia/pix2gestalt#inference-and-weights).
- Beyond amodal perception, our repository can also be used to fine-tune Stable Diffusion in an image-conditioned manner with spatial prompts, such as binary masks.
- pix2gestalt was accepted to CVPR 2024, available on [arXiv](https://arxiv.org/abs/2401.14398)!

##  Installation
```
conda create -n pix2gestalt python=3.9
conda activate pix2gestalt
cd pix2gestalt
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```
Note: We tested the installation processes on a system with Ubuntu 20.04 with NVIDIA GPUs using Ampere architecture. 

## Inference and Weights

First, download the pix2gestalt weights under `pix2gestalt/ckpt` through one of the following sources:

```
https://huggingface.co/cvlab/pix2gestalt-weights/tree/main

wget -c -P ./ckpt https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt
```
Note that we have released 2 model weights: epoch=000005.ckpt and epoch=000010.ckpt. By default, we use epoch=000005.ckpt which is the checkpoint after finetuning for 5 epochs on our [dataset](https://github.com/cvlab-columbia/pix2gestalt?tab=readme-ov-file#dataset). We have also released epoch=000010.ckpt, trained for 10 epochs. This checkpoint can be desirable for synthetic occlusion settings (given our dataset approach), though it may naturally suffer in zero-shot generalization compared to our default model.

Download [SAM](https://segment-anything.com/) checkpoints:
```
wget -c -P ./ckpt https://gestalt.cs.columbia.edu/assets/sam_vit_{b,h,l}.pth
```

Run our Gradio demo for amodal completion and segmentation:

```
python app.py
```

Note that this app uses 22-28 GB of VRAM, so it may not be possible to run it on any GPU.

For inference without the Gradio demo, we provide standalone functionality for each component [here](./pix2gestalt/inference.py), encapsulated by the [run_pix2gestalt](./pix2gestalt/inference.py#L138) method. It supports both predicted modal masks from SAM (like our demo) or ground truth modal masks. 

## 📊 Evaluation on BSDS-A Benchmark

We provide an **all-in-one automated pipeline** ([`run_eval_bsds.sh`](./run_eval_bsds.sh)) and a standalone evaluation script ([`eval_bsds.py`](./eval_bsds.py)) to evaluate amodal mask completion on the **BSDS-A (Berkeley Segmentation Dataset - Amodal)** benchmark.

### ⚡ Quick Start (1-Command All-in-One Execution)

The bash script `run_eval_bsds.sh` is 100% self-sufficient and handles everything automatically:
1. **Environment Setup:** Automatically detects and activates your environment (Conda, cloud `/venv/pix2gestalt`, RunPod, Vast.ai, etc.).
2. **GPU & PyTorch Compatibility:** Automatically verifies CUDA and upgrades PyTorch to CUDA 12.1 (supports modern Ada Lovelace RTX 40-series `sm_89`, Hopper `sm_90`, and Ampere).
3. **Weight Management:** Automatically downloads and verifies pretrained weights (`epoch=000005.ckpt`, ~15.4 GB with resume capability).
4. **Data Preparation:** Automatically downloads and extracts BSDS500 test images to `data/bsds/images/`.
5. **Annotation Fallback:** Automatically switches to synthetic amodal benchmarking on real BSDS500 images (`--auto_synth_ann`) if official annotations are not yet placed.
6. **Execution & Metrics:** Runs DDIM sampling, segments amodal masks via thresholding, and computes **Amodal mIoU** with full JSON and TXT summary reports.

Simply run:
```bash
bash run_eval_bsds.sh
```

#### Handy Execution Presets:

- **Quick Smoke Test (2 dummy instances)** — verifies GPU inference, autocast, and diffusion sampling in ~10 seconds:
  ```bash
  bash run_eval_bsds.sh --dummy_test
  ```

- **Fast Profiling (first 10 instances, 50 DDIM steps)**:
  ```bash
  bash run_eval_bsds.sh --max_samples 10 --ddim_steps 50 --n_samples 1
  ```

- **Full Benchmark on 200 BSDS500 Images** (synthetic amodal occlusions):
  ```bash
  bash run_eval_bsds.sh --auto_synth_ann --ddim_steps 50
  ```

---

### 📂 Dataset & Annotations

- **BSDS500 Images:** Automatically downloaded and placed in `data/bsds/images/` by `run_eval_bsds.sh`.
- **BSDS-A Annotations (`BSDS_amodal_test.json`):**
  - **Official Ground Truth:** If you have requested and received the official `BSDS_amodal_test.json` from the original authors (Zhu et al., CVPR 2017), place it in:
    ```
    data/bsds/annotations/BSDS_amodal_test.json
    ```
    The script will automatically detect and evaluate against the official ground truth.
  - **Automatic Synthetic Benchmark:** If the official annotation file is not present, `run_eval_bsds.sh` automatically enables `--auto_synth_ann` to generate reproducible synthetic amodal masks directly on the 200 real BSDS500 images.

---

### ⚙️ Standalone Python Evaluation (`eval_bsds.py`)

You can also run the evaluation script directly using Python:

```bash
python eval_bsds.py \
    --config pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml \
    --ckpt pix2gestalt/ckpt/epoch=000005.ckpt \
    --dataset_dir data/bsds \
    --output_dir results \
    --ddim_steps 50 \
    --guidance_scale 2.0 \
    --n_samples 1 \
    --device cuda
```

#### Key Arguments:
| Argument | Default | Description |
| :--- | :---: | :--- |
| `--config` | `...` | Path to model YAML configuration |
| `--ckpt` | `...` | Path to checkpoint file (`epoch=000005.ckpt`) |
| `--dataset_dir` | `data/bsds` | Path to dataset root directory |
| `--ann_file` | `None` | Path to `BSDS_amodal_test.json` (auto-detected if inside `dataset_dir`) |
| `--ddim_steps` | `50` | Number of DDIM diffusion denoising steps |
| `--guidance_scale` | `2.0` | Classifier-free guidance scale |
| `--n_samples` | `1` | Number of synthesized completions generated per object |
| `--threshold` | `240` | White background threshold (0–255) for amodal mask segmentation |
| `--device` | `cuda` | Target compute device (`cuda` or `cpu`) |
| `--max_samples` | `None` | Limit evaluation to the first $N$ instances |
| `--dummy_test` | `False` | Run smoke test on 2 synthetic test instances |
| `--auto_synth_ann` | `False` | Synthesize amodal masks on real BSDS500 test images |
| `--save_visualizations`| `True` | Save visual comparisons (Input, Visible, Completed, Pred Mask, GT Mask) |

#### Results & Output:
Evaluation results and visual comparisons are saved to `results/`:
- `results/summary_metrics.json`: Overall Amodal mIoU, IoU distribution, duration, and parameter log.
- `results/summary_report.txt`: Clean, human-readable summary table.
- `results/eval_results.json`: Per-instance IoU scores and metadata.
- `results/visualizations/`: Side-by-side composite images comparing input, visible mask, amodal generation, and ground-truth masks.

### Training
Download the image-conditioned Stable Diffusion checkpoint released by Lambda Labs: 

```
wget -c -P ./ckpt https://gestalt.cs.columbia.edu/assets/sd-image-conditioned-v2.ckpt
```

Then, download our fine-tuning dataset via the instructions [here](https://github.com/cvlab-columbia/pix2gestalt?tab=readme-ov-file#dataset) and update its path (see `data:params:root_dir`) in our [config](./pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml).

Run training command:  
```
python main.py \
    -t \
    --base configs/sd-finetune-pix2gestalt-c_concat-256.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 2 \
    --finetune_from ckpt/sd-image-conditioned-v2.ckpt
```
Note that this training script is set for an 8-GPU system, each with 80GB of VRAM. Empirically, the large batch size is very important for "stably" fine-tuning Stable Diffusion in an image conditioned manner. If you have smaller GPUs, consider using smaller batch sizes with gradient accumulation to obtain a similar effective batch size.

### Dataset
Download and extract our dataset of occluded objects & their whole counterparts with:
```
wget https://gestalt.cs.columbia.edu/assets/pix2gestalt_occlusions_release.tar.gz

tar -xvf pix2gestalt_occlusions_release.tar.gz
```
Disclaimer: note that the source images are from the [Segment Anything-1B Dataset](https://segment-anything.com/dataset/index.html), which has faces and license plates de-identified. For amodal perception targeted specifically for such domains, we recommend re-training or fine-tuning pix2gestalt via our custom trainining instructions. 

The dataset is intended for research purposes only. The licenses for the source images are released under the same license that they are in SA-1B.

### Amodal Recognition and 3D Reconstruction
Since we synthesize RGB images of whole objects (amodal completion), our approach makes it straightforward to equip various computer vision methods with the ability to handle occlusions, beyond amodal segmentation.

For recognition, we use [CLIP](https://github.com/openai/CLIP) as the base open-vocabulary classifier. For novel view synthesis and  3D reconstruction, we use [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer). Refer to our [paper](https://gestalt.cs.columbia.edu/static/pix2gestalt.pdf) and [supplementary](https://gestalt.cs.columbia.edu/static/supplementary.pdf) for more details.


## Citation
If you use this code, please consider citing the paper as:
```
@article{ozguroglu2024pix2gestalt,
        title={pix2gestalt: Amodal Segmentation by Synthesizing Wholes},
        author={Ege Ozguroglu and Ruoshi Liu and D\'idac Sur\'s and Dian Chen and Achal Dave and Pavel Tokmakov and Carl Vondrick},
        journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2024}
}
```

##  Acknowledgement
This research is based on work partially supported by the Toyota Research Institute, the DARPA MCS program under Federal Agreement No. N660011924032, the NSF NRI Award \#1925157, and the NSF AI Institute for Artificial and Natural Intelligence Award \#2229929. DS is supported by the Microsoft PhD Fellowship.
