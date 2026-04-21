# AI6126 - Output space UDA and MLLM Transfer

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Ardacandra/ai6126_output_space_uda_and_mllm_transfer.git
cd ai6126_output_space_uda_and_mllm_transfer
```

### 2. Set Up the Conda Environment

```bash
# Create a new conda environment with Python 3.9
conda create -n ai6126_output_space_uda_and_mllm_transfer python=3.9 -y

# Activate the environment
conda activate ai6126_output_space_uda_and_mllm_transfer

# Install PyTorch with CUDA 12.1 support for GPU acceleration
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install all other project dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Prepare Dataset

The project expects datasets inside the `dataset/` directory.

### Downloaded Datasets

- Office-31  
	Reference link: `https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#office-31`  
	Local path: `dataset/office31/`  

- Office-Home  
	Reference link: `https://www.hemanthdv.org/officeHomeDataset.html`  
	Local path: `dataset/OfficeHomeDataset/`  

- PACS  
	Reference link: `https://www.kaggle.com/datasets/ma3ple/pacs-dataset`  
	Local path: `dataset/pacs/`  

- MNIST (via torchvision)  
	Reference link: `https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html`  
	Local path: `dataset/MNIST/`  

- SVHN (via torchvision)  
	Reference link: `https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html`  
	Local path: `dataset/SVHN/`  

- USPS (via torchvision)  
	Reference link: `https://pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html`  
	Local path: `dataset/USPS/`  

### Download MNIST, SVHN, USPS from PyTorch

Run:

```bash
python scripts/download_torch_datasets.py
```

This script downloads train/test splits into dataset-specific folders:

- `dataset/MNIST/`
- `dataset/SVHN/`
- `dataset/USPS/`

### Verify All Datasets

Expected directory tree:

```text
dataset/
├── office31/
│   ├── amazon/
│   ├── dslr/
│   └── webcam/
├── OfficeHomeDataset/
│   ├── Art/
│   ├── Clipart/
│   ├── Product/
│   └── Real World/
├── pacs/
│   ├── art_painting/
│   ├── cartoon/
│   ├── photo/
│   └── sketch/
├── MNIST/
├── SVHN/
└── USPS/
```

After downloading or placing all datasets, verify they are loaded correctly:

```bash
python scripts/visualize_dataset.py
```