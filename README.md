# CAI-Diffusion
CAI-Diff: Causal Anticipation and Interaction Diffusion for Facial Reaction Generation


## 🛠️ Dependency Installation

We provide detailed instructions for setting up the environment using conda. First, create and activate a new environment:
``` shell
conda create -n react python=3.10
conda activate react
```

### 1. Install PyTorch
First, check your CUDA version:
``` shell
nvidia-smi
```
Visit [Pytorch official website](https://pytorch.org/) to get the appropriate installation command. For example:
``` shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install PyTorch3D Dependencies
Install the following dependencies:
``` shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
For CUDA versions older than 11.7, you will need to install the CUB library. 
``` shell
conda install -c bottler nvidiacub
```

### 3. Install PyTorch3D
First, verify your CUDA version in Python:
``` shell
import torch
torch.version.cuda
```
[//]: # (Download `pytorch3d` file based on the version of python, cuda and pytorch from https://anaconda.org/pytorch3d/pytorch3d/files. For example, to install for Python 3.8, PyTorch 1.12.1 and CUDA 11.6, select the below file to download)
Download the appropriate `PyTorch3D` package from [Anaconda](https://anaconda.org/pytorch3d/pytorch3d/files) based on your Python, CUDA, and PyTorch versions. For example, for Python 3.10, CUDA 11.6, and PyTorch 1.12.0:

[//]: # (Finally install `pytorch3d` via the downloaded `.tar.bz2` file via conda)
``` shell
# linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
conda install linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
```

### 4. Install Additional Dependencies
[//]: # (pip install omegaconf scikit-video pandas soundfile av decord tensorboard numpy tslearn scikit-image matplotlib imageio plotly opencv-python librosa einops)
Install all remaining dependencies specified in requirements.txt:
``` shell
pip install -r requirements.txt
```
