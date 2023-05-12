
# URDT
# Unlimited-Resolution Diffusion Transformer: A Patch-Cycle Diffusion Model for Progressive Image Restoration
<!By Liyan Wang, Qinyu Yang, Bo Fu, Ximing Li, and Zhixun Su>

> **Abstract:** *Single image restoration (IR) methods trained on synthetic data generalize poorly to complex and diverse real data. Recently, diffusion models (DM) have achieved state-of-the-art performance by gradually generating high-quality and diverse data samples with fixed resolution from noisy data, but it is impractical to apply to the IR problems of arbitrary resolution. Our main proposal is to create a progressive restoration process based on DM, to achieve high-quality restoration of any resolution degraded images. Therefore, we propose a novel Unlimited-Resolution Diffusion Transformer (URDT) for progressive IR. Specifically, our URDT owns a coarse-to-fine training pipeline. The coarse training pipeline adopts the training manner of the conditional diffusion model to train the model, while the fine training pipeline further trains by selecting few sampling steps to construct a progressive restoration process. Moreover, the noise estimation network of our URDT is designed as a novel Transformer-based U-shape architecture named DT, which processes unlimited-resolution images by redesigning Diffusion Transformer Blocks (DTBs) with time embedding and Multi-Dconv Head Transposed Attention (MDTA). In this case, a patch-cycle diffusion training strategy is designed to make DT more conducive to adapting different image resolutions. Extensive experiments show that the proposed URDT demonstrates strong generalization capabilities on benchmark datasets for image deraining, image motion deblurring, and real image denoising. By leveraging the advantages of diffusion models, URDT offers a promising new approach to image restoration tasks.* 
<hr />

# Updating!!!

## Coarse Training Pipeline and DT Network

<img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/fig1.png#pic_center"> 

## Fine Training Pipeline and Sampling

<img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/fig2.png#pic_center"> 

## Requirements
- CUDA 10.1 (or later)
- Python 3.7 (or later)
- Pytorch 1.8.1 (or later)
- Torchvision 0.19
- OpenCV 4.7.0
- tensorboard, skimage, scipy, lmdb, tqdm, yaml, einops, natsort

## Training and Evaluation

Training and Testing instructions for Image Deraining, Image Motion Deblurring, and Real Image Denoising are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="center">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
    <th align="center">URDT's Visual Results</th>
  </tr>
  <tr>
    <td align="center">Image Deraining</td>
    <td align="center"><a href="Deraining/README.md#Training">Link</a></td>
    <td align="center"><a href="Deraining/README.md#Testing">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1v4aAFDAojHtedtRmPcqVKJcAixW5dZ8m">Download</a></td>
  </tr>
  <tr>
    <td align="center">Image Motion Deblurring</td>
    <td align="center"><a href="Motion_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Motion_Deblurring/README.md#Testing">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1qYVPblP0kCyfIoxDQ2NBsdbv_MoZ24S4">Download</a></td>
  </tr>
  <tr>
     <td align="center">Real ImageDenoising</td>
    <td align="center"><a href="Denoising/README.md#Training">Link</a></td>
    <td align="center"><a href="Denoising/README.md#Testing">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1hgSYcwSLktFh42LA9bDXTLUuNzThdJVA">Download</a></td>
  </tr>
</table>

## Experimental Results

<details>
<summary><strong>Image Deraining</strong> (click to expand) </summary>

<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/tab1.png#pic_center"></p> 
<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/fig3.png#pic_center" width="1000"></p> 

</details>

<details>
<summary><strong>Image Motion Deblurring</strong> (click to expand) </summary>

<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/tab2.png#pic_center" width="500"></p>
<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/fig4.png#pic_center" width="1000"></p>
</details>

<details>
<summary><strong>Real Image Denoising</strong> (click to expand) </summary>

<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/tab3.png#pic_center" width="500"></p>
<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/fig5.png#pic_center" width="1000"></p>
<p align="center"><img src = "https://github.com/wlydlut/URDT-main/blob/main/Figs/fig6.png#pic_center" width="1000"></p>
</details>

## Contact
<!Should you have any questions, please contact wangliyan@mail.dlut.edu.cn >


**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox and [Restormer](https://github.com/swz30/Restormer). 

