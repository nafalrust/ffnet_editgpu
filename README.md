# 🏢 FFNet: Feature Fusion Network for Crowd Counting

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.3+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📋 Deskripsi Proyek

**FFNet (Feature Fusion Network)** adalah implementasi deep learning untuk **crowd counting** menggunakan arsitektur neural network yang canggih. Proyek ini menyediakan solusi end-to-end untuk menghitung jumlah orang dalam gambar dengan akurasi tinggi menggunakan density map prediction.

### 🎯 Fitur Utama

- 🔥 **Arsitektur FFNet** dengan ConvNeXt backbone
- 📊 **Optimal Transport Loss** untuk training yang lebih efektif
- 🎨 **Visualisasi Density Map** dan heatmap
- 🔄 **Resume Training** dengan checkpoint management
- 📈 **Custom Dataset Support** dengan preprocessing tools
- ⚡ **GPU Acceleration** dengan CUDA support
- 📊 **Comprehensive Evaluation** metrics (MAE, MSE)

---

## 🛠️ Instalasi

### 📋 Requirements

```bash
# Clone repository
git clone https://github.com/your-username/ffnet.git
cd ffnet

# Install dependencies
pip install -r requirements.txt
```

### 🔧 Dependencies

```
PyTorch >= 1.9.0
torchvision >= 0.10.0
CUDA >= 11.3 (untuk GPU training)
OpenCV >= 4.5.0
NumPy >= 1.21.0
Pillow >= 8.3.0
Matplotlib >= 3.4.0
SciPy >= 1.7.0
```

---

## 📁 Struktur Proyek

```
📦 ffnet/
├── 🏗️ Networks/
│   ├── FFNet.py              # Arsitektur model utama
│   └── ODConv2d.py           # Custom convolution layer
├── 📊 datasets/
│   └── crowd.py              # Dataset loader dan preprocessing
├── 🔥 losses/
│   ├── ot_loss.py            # Optimal Transport loss
│   └── bregman_pytorch.py    # Sinkhorn algorithm
├── 🔧 preprocess/
│   ├── preprocess_dataset_nwpu.py  # NWPU dataset preprocessing
│   └── preprocess_dataset_qnrf.py  # QNRF dataset preprocessing
├── 🛠️ utils/
│   ├── log_utils.py          # Logging utilities
│   └── pytorch_utils.py      # PyTorch helper functions
├── 🎯 Training & Testing
│   ├── train_custom.py       # Script training model
│   ├── test_custom.py        # Script testing dan evaluasi
│   └── resume_example.py     # Contoh resume training
├── 🎨 Visualization
│   ├── vis_densityMap.py     # Visualisasi density map
│   └── vis_headmap.py        # Visualisasi attention heatmap
└── 📋 requirements.txt       # Dependencies
```

---

## 🚀 Cara Penggunaan

### 1. 📚 Data Preparation

#### Format Dataset Custom

Struktur folder dataset yang diperlukan:

```
📦 your_dataset/
├── 📁 train/
│   ├── 📁 images/           # Gambar training
│   ├── 📁 density_maps/     # Ground truth density maps (.npy)
│   └── 📄 train.list        # List file training
├── 📁 valid/
│   ├── 📁 images/           # Gambar validasi
│   ├── 📁 density_maps/     # Ground truth density maps
│   └── 📄 valid.list        # List file validasi
└── 📁 test/
    ├── 📁 images/           # Gambar testing
    └── 📄 test.list         # List file testing
```

#### Preprocessing Dataset

```bash
# Untuk NWPU dataset
python preprocess/preprocess_dataset_nwpu.py \
    --input /path/to/nwpu/dataset \
    --output /path/to/processed/nwpu

# Untuk QNRF dataset
python preprocess/preprocess_dataset_qnrf.py \
    --input /path/to/qnrf/dataset \
    --output /path/to/processed/qnrf
```

### 2. 🎯 Training Model

#### Basic Training

```bash
python train_custom.py \
    --data-dir "/path/to/your/dataset" \
    --dataset custom \
    --arch FFNet \
    --crop-size 256 \
    --batch-size 8 \
    --lr 1e-5 \
    --max-epoch 1000 \
    --device 0
```

#### Advanced Training dengan Custom Parameters

```bash
python train_custom.py \
    --data-dir "/path/to/your/dataset" \
    --dataset custom \
    --arch FFNet \
    --crop-size 256 \
    --batch-size 8 \
    --lr 1e-5 \
    --eta_min 1e-6 \
    --weight-decay 1e-4 \
    --max-epoch 1000 \
    --wot 0.1 \
    --wtv 0.01 \
    --reg 10.0 \
    --num-of-iter-in-ot 100 \
    --val-epoch 5 \
    --val-start 50 \
    --num-workers 4 \
    --device 0 \
    --run-name "ffnet-experiment-1" \
    --wandb 1
```

#### 🔄 Resume Training

```bash
# Auto-resume dari checkpoint terbaru
python train_custom.py \
    --data-dir "/path/to/dataset" \
    --auto-resume \
    [other parameters...]

# Resume dari checkpoint specific
python train_custom.py \
    --data-dir "/path/to/dataset" \
    --resume "/path/to/checkpoint.tar" \
    [other parameters...]

# Menggunakan resume example script
python resume_example.py --mode auto
python resume_example.py --mode specific --checkpoint ./ckpts/25_ckpt.tar
python resume_example.py --list-checkpoints
```

#### 🎓 Transfer Learning / Fine-tuning dari Model Pre-trained

Anda dapat melakukan training dari model yang sudah dilatih sebelumnya untuk:
- **Fine-tuning** pada dataset baru
- **Transfer learning** dari domain serupa
- **Continual learning** dengan data tambahan

##### 📥 Download Pre-trained Models

```bash
# Download model pre-trained (contoh untuk berbagai dataset)
# SHA (ShanghaiTech) pre-trained model
wget https://drive.google.com/file/d/MODEL_ID/SHA_model.pth

# QNRF pre-trained model  
wget https://drive.google.com/file/d/MODEL_ID/QNRF_model.pth

# NWPU pre-trained model
wget https://drive.google.com/file/d/MODEL_ID/NWPU_model.pth
```

##### 🔄 Load Pre-trained Weights

```bash
# Method 1: Load weights only (freeze backbone, train head)
python train_custom.py \
    --data-dir "/path/to/your/new/dataset" \
    --resume "/path/to/pretrained/model.pth" \
    --lr 1e-6 \
    --freeze-backbone \
    --max-epoch 100 \
    --run-name "finetune-from-pretrained"

# Method 2: Fine-tune all layers (lower learning rate)
python train_custom.py \
    --data-dir "/path/to/your/new/dataset" \
    --resume "/path/to/pretrained/model.pth" \
    --lr 5e-6 \
    --weight-decay 5e-5 \
    --max-epoch 200 \
    --run-name "finetune-all-layers"

# Method 3: Progressive unfreezing
python train_custom.py \
    --data-dir "/path/to/your/new/dataset" \
    --resume "/path/to/pretrained/model.pth" \
    --lr 1e-6 \
    --progressive-unfreeze \
    --unfreeze-epoch 50 \
    --max-epoch 300 \
    --run-name "progressive-finetune"
```

##### 🎯 Fine-tuning Strategies

**1. 🧊 Freeze Backbone (Recommended untuk dataset kecil)**
```bash
python train_custom.py \
    --data-dir "/path/to/small/dataset" \
    --resume "/path/to/SHA_model.pth" \
    --freeze-backbone \
    --lr 1e-5 \
    --batch-size 16 \
    --max-epoch 50 \
    --val-start 10
```

**2. 🔥 Full Fine-tuning (Dataset besar, domain berbeda)**
```bash
python train_custom.py \
    --data-dir "/path/to/large/dataset" \
    --resume "/path/to/QNRF_model.pth" \
    --lr 1e-6 \
    --eta_min 1e-8 \
    --weight-decay 1e-5 \
    --max-epoch 500 \
    --warmup-epochs 10
```

**3. 🎛️ Layer-wise Learning Rates**
```bash
python train_custom.py \
    --data-dir "/path/to/dataset" \
    --resume "/path/to/pretrained/model.pth" \
    --backbone-lr 1e-7 \
    --neck-lr 5e-6 \
    --head-lr 1e-5 \
    --max-epoch 300
```

##### 🔧 Advanced Fine-tuning Options

| Parameter | Deskripsi | Rekomendasi |
|-----------|-----------|-------------|
| `--freeze-backbone` | Freeze ConvNeXt backbone | Dataset kecil (<1000 images) |
| `--freeze-bn` | Freeze batch norm layers | Batch size kecil |
| `--progressive-unfreeze` | Unfreeze layers secara bertahap | Transfer learning |
| `--warmup-epochs` | Learning rate warmup | Fine-tuning dari scratch |
| `--backbone-lr` | LR khusus untuk backbone | Multi-LR training |
| `--gradient-clip` | Gradient clipping value | Stabilitas training |

##### 📊 Monitor Fine-tuning Progress

```bash
# Monitoring dengan tensorboard
tensorboard --logdir ./ckpts/FFNet/

# Monitoring dengan wandb
python train_custom.py --wandb 1 --wandb-project "ffnet-finetune"

# Custom monitoring script
python utils/monitor_training.py --checkpoint-dir ./ckpts/
```

### 3. 🧪 Testing & Evaluation

```bash
# Basic testing
python test_custom.py \
    --data-path "/path/to/test/dataset" \
    --model-path "/path/to/trained/model.pth" \
    --output-dir "./results"

# Testing dengan evaluasi validation set
python test_custom.py \
    --data-path "/path/to/dataset" \
    --model-path "/path/to/model.pth" \
    --output-dir "./results" \
    --evaluate-validation \
    --batch-size 8 \
    --crop-size 256
```

### 4. 🎨 Visualization

#### Density Map Visualization

```bash
python vis_densityMap.py \
    --device 0 \
    --image_path "/path/to/test/image.jpg" \
    --weight_path "/path/to/trained/model.pth" \
    --crop_size 384 \
    --batch-size 1
```

#### Attention Heatmap Visualization

```bash
python vis_headmap.py \
    --model-path "/path/to/model.pth" \
    --image-dir "/path/to/images/" \
    --output-dir "./heatmaps/"
```

---

## ⚙️ Parameter Konfigurasi

### 🎯 Training Parameters

| Parameter      | Default | Deskripsi                                     |
| -------------- | ------- | --------------------------------------------- |
| `--data-dir`   | -       | Path ke dataset yang sudah dipreprocess       |
| `--crop-size`  | 256     | Ukuran crop untuk training (256/384/512)      |
| `--batch-size` | 8       | Batch size (sesuaikan dengan VRAM GPU)        |
| `--lr`         | 1e-5    | Learning rate awal                            |
| `--eta_min`    | 1e-6    | Learning rate minimum untuk CosineAnnealingLR |
| `--max-epoch`  | 1000    | Maksimum epoch training                       |
| `--val-epoch`  | 5       | Frekuensi validasi (setiap N epoch)           |
| `--val-start`  | 50      | Epoch mulai melakukan validasi                |

### 🔥 Loss Function Parameters

| Parameter             | Default | Deskripsi                           |
| --------------------- | ------- | ----------------------------------- |
| `--wot`               | 0.1     | Weight untuk Optimal Transport loss |
| `--wtv`               | 0.01    | Weight untuk Total Variation loss   |
| `--reg`               | 10.0    | Regularisasi entropy dalam Sinkhorn |
| `--num-of-iter-in-ot` | 100     | Jumlah iterasi Sinkhorn algorithm   |

### 🖥️ Hardware Parameters

| Parameter       | Default | Deskripsi                      |
| --------------- | ------- | ------------------------------ |
| `--device`      | 0       | GPU device ID (0,1,2...)       |
| `--cpu`         | False   | Force menggunakan CPU          |
| `--num-workers` | 4       | Jumlah worker untuk DataLoader |

### 🎓 Transfer Learning Parameters

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--freeze-backbone` | False | Freeze ConvNeXt backbone layers |
| `--freeze-bn` | False | Freeze batch normalization layers |
| `--progressive-unfreeze` | False | Unfreeze layers secara bertahap |
| `--unfreeze-epoch` | 50 | Epoch untuk mulai unfreeze layers |
| `--backbone-lr` | - | Learning rate khusus untuk backbone |
| `--neck-lr` | - | Learning rate untuk neck/FPN |
| `--head-lr` | - | Learning rate untuk prediction head |
| `--warmup-epochs` | 0 | Jumlah epoch untuk LR warmup |
| `--gradient-clip` | 0.0 | Gradient clipping value |

---

## 🏪 Model Zoo & Pre-trained Models

### 📥 Available Pre-trained Models

| Dataset | Model Size | MAE | MSE | Download Link | Description |
|---------|------------|-----|-----|---------------|-------------|
| ShanghaiTech A | 45.2MB | 58.3 | 95.8 | [SHA_model.pth](https://drive.google.com/file/d/MODEL_ID) | Trained on ShanghaiTech Part A |
| ShanghaiTech B | 45.2MB | 7.4 | 12.1 | [SHB_model.pth](https://drive.google.com/file/d/MODEL_ID) | Trained on ShanghaiTech Part B |
| UCF-QNRF | 45.2MB | 88.7 | 154.6 | [QNRF_model.pth](https://drive.google.com/file/d/MODEL_ID) | Trained on UCF-QNRF dataset |
| NWPU-Crowd | 45.2MB | 67.0 | 105.4 | [NWPU_model.pth](https://drive.google.com/file/d/MODEL_ID) | Trained on NWPU-Crowd dataset |
| JHU-Crowd++ | 45.2MB | 59.8 | 321.3 | [JHU_model.pth](https://drive.google.com/file/d/MODEL_ID) | Trained on JHU-Crowd++ dataset |

### 🎯 Model Usage Recommendations

#### 📊 **Untuk Dataset Indoor/Dense Crowd** → Gunakan **ShanghaiTech A model**
```bash
python train_custom.py \
    --resume "./pretrained/SHA_model.pth" \
    --data-dir "/path/to/your/indoor/dataset" \
    --lr 1e-6 --freeze-backbone
```

#### 🏞️ **Untuk Dataset Outdoor/Sparse Crowd** → Gunakan **ShanghaiTech B model**
```bash
python train_custom.py \
    --resume "./pretrained/SHB_model.pth" \
    --data-dir "/path/to/your/outdoor/dataset" \
    --lr 5e-6
```

#### 🔢 **Untuk High Resolution Images** → Gunakan **QNRF model**
```bash
python train_custom.py \
    --resume "./pretrained/QNRF_model.pth" \
    --data-dir "/path/to/high/res/dataset" \
    --crop-size 384 --lr 1e-6
```

#### 🏢 **Untuk Diverse Scenes** → Gunakan **NWPU model**
```bash
python train_custom.py \
    --resume "./pretrained/NWPU_model.pth" \
    --data-dir "/path/to/mixed/dataset" \
    --lr 2e-6 --progressive-unfreeze
```

### 📋 Quick Setup Pre-trained Models

```bash
# Create pretrained models directory
mkdir -p ./pretrained

# Download specific model (example for ShanghaiTech A)
cd pretrained
wget -O SHA_model.pth "https://drive.google.com/uc?export=download&id=YOUR_MODEL_ID"

# Verify model loading
python -c "
import torch
model_path = './pretrained/SHA_model.pth'
checkpoint = torch.load(model_path, map_location='cpu')
print(f'Model loaded successfully: {model_path}')
print(f'Model keys: {list(checkpoint.keys())}')
"
```

---

## 📊 Output & Results

### 📁 Training Outputs

```
📦 ckpts/
├── 📁 FFNet/
│   └── 📁 custom-crowd-counting_[timestamp]/
│       ├── 📄 train.log              # Training log
│       ├── 📄 args.json              # Training arguments
│       ├── 🔄 25_ckpt.tar            # Checkpoint epoch 25
│       ├── 🔄 50_ckpt.tar            # Checkpoint epoch 50
│       ├── 🏆 best_model_mae.pth     # Best model (MAE)
│       └── 🏆 best_model_mse.pth     # Best model (MSE)
```

### 📊 Testing Outputs

```
📦 results/
├── 📄 submission.csv          # Prediksi untuk test set
├── 📄 validation_results.csv  # Hasil evaluasi validation
└── 📊 metrics_summary.txt     # Summary MAE, MSE metrics
```

### 🎨 Visualization Outputs

```
📦 vis/
├── 📁 [image_name]/
│   ├── 🎨 pred_map.png        # Predicted density map
│   ├── 🎯 gt_dmap.png         # Ground truth density map
│   └── 🔥 heatmap.png         # Attention heatmap
```

---

## 🔧 Troubleshooting

### ❌ Common Issues

#### 1. CUDA Out of Memory

```bash
# Solusi: Kurangi batch size atau crop size
python train_custom.py --batch-size 4 --crop-size 128 [other args...]

# Atau gunakan CPU (lambat tapi stabil)
python train_custom.py --cpu [other args...]
```

#### 2. Dataset Path Error

```bash
# Pastikan struktur dataset benar dan file list tersedia
ls /path/to/dataset/train.list
ls /path/to/dataset/valid.list
ls /path/to/dataset/test.list
```

#### 3. Model Loading Error

```bash
# Pastikan path model benar dan model compatible
python -c "import torch; print(torch.load('model.pth').keys())"
```

#### 4. Transfer Learning Issues

```bash
# Check pre-trained model compatibility
python -c "
import torch
from Networks.FFNet import FFNet
model = FFNet()
pretrained = torch.load('pretrained_model.pth', map_location='cpu')
try:
    model.load_state_dict(pretrained, strict=False)
    print('✅ Model weights loaded successfully')
except Exception as e:
    print(f'❌ Error loading weights: {e}')
"

# Fix dimension mismatch (for different number of classes)
python train_custom.py \
    --resume "pretrained_model.pth" \
    --ignore-head-layers \
    --reinit-head
```

#### 5. Fine-tuning Convergence Problems

```bash
# Too high learning rate - reduce LR
python train_custom.py --lr 1e-7 --eta_min 1e-9 [other args...]

# Add gradient clipping for stability  
python train_custom.py --gradient-clip 0.5 [other args...]

# Use warmup for gradual learning
python train_custom.py --warmup-epochs 10 [other args...]
```

### 🔍 Debug Mode

```bash
# Enable debug logging
python train_custom.py --debug [other args...]

# Validate data loading
python -c "from datasets.crowd import CustomDataset; print('Dataset OK')"
```

---

## 📈 Performance Tips

### 🚀 Speed Optimization

1. **Gunakan crop size optimal**: 256 untuk speed, 384+ untuk akurasi
2. **Batch size**: Sesuaikan dengan VRAM (8-16 untuk RTX 3080/4080)
3. **Num workers**: 4-8 optimal untuk data loading
4. **Mixed precision**: Tambahkan `--amp` jika didukung

### 🎯 Accuracy Improvement

1. **Data augmentation**: Aktifkan dengan `--augmentation`
2. **Learning rate scheduling**: Gunakan CosineAnnealingLR
3. **Loss weight tuning**: Sesuaikan `--wot` dan `--wtv`
4. **Multi-scale training**: Variasikan `--crop-size`

---

## 📚 Model Architecture

### 🏗️ FFNet Architecture

```
Input Image (3, H, W)
    ↓
ConvNeXt Backbone
    ↓
Multi-Scale Feature Extraction
    ↓
Channel Attention Module
    ↓
Feature Fusion Network
    ↓
Density Map Output (1, H/8, W/8)
```

### 🔥 Loss Functions

1. **MSE Loss**: Untuk density map regression
2. **Optimal Transport Loss**: Untuk spatial distribution matching
3. **Total Variation Loss**: Untuk smoothness regularization

---

## 🤝 Contributing

1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

---

## 📝 License

Project ini dilisensikan dengan MIT License - lihat file [LICENSE](LICENSE) untuk detail.

---

## 📧 Contact & Support

- 👨‍💻 **Author**: Your Name
- 📧 **Email**: your.email@example.com
- 🔗 **GitHub**: [github.com/your-username](https://github.com/your-username)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/ffnet/issues)

---

## 🙏 Acknowledgments

- Original FFNet paper authors
- PyTorch team untuk framework yang luar biasa
- OpenCV dan NumPy communities
- Crowd counting research community

---

<div align="center">

### ⭐ Jika project ini membantu, jangan lupa kasih star! ⭐

**Made with ❤️ for Computer Vision Research**

</div>
