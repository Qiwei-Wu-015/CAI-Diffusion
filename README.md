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

## 👨‍🏫 Get Started 

<details><summary> <b> Data </b> </summary>
<p>
 
**Challenge Data Description ([Homepage](https://sites.google.com/cam.ac.uk/react2024)):**

We divided the datasets into training, test, and validation sets following an estimated 60%/20%/20% splitting ratio. Specifically, we split the datasets with a subject-independent strategy (i.e., the same subject was never included in the train and test sets).

[//]: # (- Dataset Directory Structure: &#40;training and validation sets are provided at this stage&#41;)
- *video-raw* folder contains raw videos (with the resolution of 1920 * 1080)
- *video-face-crop* folder contains face-cropped videos (with the resolution of 384 * 384)
- *facial-attributes* folder contains sequences of frame-level 25-dimension facial attributes (15 AUs’ occurrences, valence and arousal intensities, and the probabilities of eight categorical facial expressions)
- *coefficients* folder contains sequences of 58-dimension (52-d expression, 3-d rotation, and 3-d translation) 3DMM coefficients extracted from corresponding videos
- *audio* folder contains wav files extracted from raw video files

Appropriate real facial reactions (Ground-Truths):
- During data recording, the semantic contexts are carefully controlled through the 23 distinct sessions (session0, session1, …, session22), each of which is guided by a few pre-defined sentences posted by the speaker. This provides a consistent session-specific context across dyadic interactions between different speakers and listeners. More specifically, for the speaker behaviour expressed in a specific session, we define all facial reactions expressed by different listeners under the same session to be appropriate facial reactions (i.e., ground-truth) for responding to it.
   
**Data organization (`./data`) is listed below:**
The example of data structure.
```

├── val
├── test
├── train
    ├── coefficients (.npy)
    ├── video-face-crop (.mp4)
    ├── video-raw (.mp4)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.mp4
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.mp4
                ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.mp4
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.mp4
                ├── ...
    ├── facial-attributes (.npy)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.npy
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.npy
                ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.npy
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.npy
                ├── ...
    ├── audio (.wav)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.wav
                ├── ...
            ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.wav
                ├── ...
            ├── ...
```

</p>
</details>

<details><summary> <b> External Tool Preparation </b> </summary>
<p>

We use 3DMM coefficients to represent a 3D listener or speaker, and for further 3D-to-2D frame rendering. The baselines leverage [3DMM model](https://github.com/LizhenWangT/FaceVerse) to extract 3DMM coefficients, and render 3D facial reactions.  

- You should first download 3DMM (FaceVerse version 2 model) at this [page](https://github.com/LizhenWangT/FaceVerse) 
 
  and then put it in the folder (`external/FaceVerse/data/`).
 
  We provide our extracted 3DMM coefficients (which are used for our baseline visualisation) at [OneDrive](https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu). 

  We also provide the `mean_face.npy` at this [OneDrive link](https://1drv.ms/u/c/4c787027becb2e91/EXhSObCHXUhHg0-Geyy4_6QB7b611XFgbJcIoGymcmzS-Q?e=NT8IKj) and `std_face.npy` at this [OneDrive link](https://1drv.ms/u/c/4c787027becb2e91/EdyIBxX-IlVEivdFxURn-BMBiK6JFSAXcp3qwCPNVboifQ?e=o5NgqM) and `reference_full.npy` at this [Onedrive link](https://1drv.ms/u/c/4c787027becb2e91/ERoBr5MNudxBgImW4jPt39sBwqFNSsvwX3OihUfU_TYpqw?e=h8mOqp) for 3DMM coefficients Data Normalization. Please download and put them in the folder (`external/FaceVerse/`).

[//]: # ( and reference_full )

Then, we use a 3D-to-2D tool [PIRender](https://github.com/RenYurui/PIRender) to render final 2D facial reaction frames.
 
- We re-trained the PIRender, and the well-trained model is provided at the [checkpoint](https://1drv.ms/u/c/4c787027becb2e91/ERLUL_QTBABHoLzCTCbUZF8Bu6e_5o0YX31rA12yv0DIcQ?e=mWKgcn). Please put it in the folder (`external/PIRender/`).

Finally, please download the compressed folder named `pretrained_models` from [this link](https://1drv.ms/u/c/4c787027becb2e91/EZ_l_EhvDbFOnmA_n69F1z0BpSqZumEcevc-iC3wVOhqhA?e=FlqhFb), and extract it into the project root directory.

</p>
</details>


<details><summary> <b> Training </b>  </summary>
<p>
 
 <b>CAI-Diffusion(based on PerFRDiff)</b>
 - Running the following shell can start training PerFRDiff baseline for the offline task: 
```shell
python main.py \
    data=motion_diffusion \
    trainer=motion_diffusion \
    trainer.batch_size=2 \
    stage=fit \
    task=offline \
    data_dir=./data
```
 &nbsp; &nbsp; or for the online task:
```shell
python main.py \
    data=motion_diffusion \
    trainer=motion_diffusion \
    trainer.batch_size=8 \
    stage=fit \
    task=online \
    data_dir=./data
```


<details><summary> <b> Pretrained weights </b>  </summary>

- [ ] to be released

</details>

<details><summary> <b> Evaluation </b>  </summary>

- [ ] to be released

</details>
