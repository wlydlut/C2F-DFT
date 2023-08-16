## Datasets

-Train datasets: [GoPro](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing)

-Test datasets:  [GoPro](https://drive.google.com/file/d/1k6DTSHu4saUgrGTYkkZXTptILyG9RRll/view?usp=sharing), [RealBlur_R](https://drive.google.com/file/d/1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS/view?usp=sharing), [RealBlur_J](https://drive.google.com/file/d/1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW/view?usp=sharing)

-Val dataset: To accelerate the training speed, we selected the first image from the GoPro test set as the validation set.

- The above dataset path is as follows
    
 `Deblurring/Datasets` <br/>
 `├──train`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
 `├──val`  <br/>
          `├──input`   <br/>
          `└──gt`   <br/>
 `└──test`  <br/>
     `├──GoPro`   <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `├──RealBlur_J`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `└──RealBlur_R` <br/>
          `├──input`   <br/>
          `└──target`  <br/>
  
## Training
First, modify the path where the project is located in the second line of the /basicsr/train.py file.

1. To train C2F-DFT in the coarse training pipeline, modify the comments on lines 129-134 and 195-237 in the /basicsr/models/image_restoration_model.py file, then run

    ```
    cd C2F-DFT-main
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt Deblurring/Options/Deblurring_C2F-DFT_Coarse.yml  --launcher pytorch
    ```

2. To train C2F-DFT in the fine training pipeline, modify the comments on lines 137-145 and 240-286 in the /basicsr/models/image_restoration_model.py file, then run

    ```
    cd C2F-DFT-main
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt Deblurring/Options/Deblurring_C2F-DFT_Fine.yml  --launcher pytorch
    ```

## Testing

1. Download the pre-trained [model](https://drive.google.com/drive/folders/1Xr6SigGj8AdvwSapqxfWWtRDU7KTPMem) and place it in `./pretrained_models/`

2. Testing
   
   Modify the path where the project is located in the second line of the Deblurring/test.py file

    #### Testing on GoPro dataset, run

    ```
    cd Deblurring
    python test.py
    ```
    #### Testing on RealBlur dataset, run

    ```
    cd Deblurring
    python test_real.py
    ```

5. Calculating PSNR/SSIM scores

    #### Calculate GoPro dataset, run

    ```
    python calculate_psnr_ssim.py
    ```
    #### Calculate RealBlur dataset, run

    ```
    python evaluate_realblur.py
    ```
