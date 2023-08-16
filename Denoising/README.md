## Datasets

-Train datasets:  [SIDD](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing)

-Test datasets:  [SIDD](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing), [DND](https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing)

-Val datasest:  [SIDD_val](https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing). To accelerate the training speed, we selected the first image from the SIDD_val test set as the validation set.
             
- Generate image patches from full-resolution training images, run
  ```
  python generate_patches_sidd.py 
  ```

- The above dataset path is as follows
    
`Denoising/Datasets` <br/>
 `├──train`  <br/>
          `├──input_crops`   <br/>
          `└──target_crops`   <br/>
 `├──val`  <br/>
          `├──input`   <br/>
          `└──gt`   <br/>
 `└──test`  <br/>
     `├──SIDD`   <br/>
          `├──ValidationNoisyBlocksSrgb.mat`   <br/>
          `└──ValidationGtBlocksSrgb.mat`   <br/>
     `├──DND`   <br/>
          `├──info.mat`   <br/>
          `└──images_srgb`   <br/>
               `├──0001.mat`   <br/>
               `├──0002.mat`   <br/>
               `├── ...    `   <br/>
               `└──0050.mat` 

## Training
First, modify the path where the project is located in the second line of the /basicsr/train.py file.

1. To train C2F-DFT in the coarse training pipeline, modify the comments on lines 129-134 and 195-237 in the /basicsr/models/image_restoration_model.py file, then run

    ```
    cd C2F-DFT-main
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt Denoising/Options/RealDenoising_C2F-DFT_Coarse.yml  --launcher pytorch
    ```

2. To train C2F-DFT in the fine training pipeline, modify the comments on lines 137-145 and 240-286 in the /basicsr/models/image_restoration_model.py file, then run

    ```
    cd C2F-DFT-main
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt Denoising/Options/RealDenoising_C2F-DFT_Fine.yml  --launcher pytorch
    ```

## Testing

- Download the pre-trained [model](https://drive.google.com/drive/folders/1jIDur6-7gob1pyq247FGRxQcZrdnUlVg) and place it in `./pretrained_models/`

#### Testing on SIDD dataset

Modify the path where the project is located in the second line of the Denoising/test_real_denoising_sidd.py file

- To obtain denoised results, run
    ```
    cd Denoising
    python test_real_denoising_sidd.py
    ```

- To reproduce PSNR/SSIM scores on SIDD data, run
    ```
    evaluate_sidd.m
    ```

#### Testing on DND dataset

Modify the path where the project is located in the second line of the Denoising/test_real_denoising_dnd.py file

- To obtain denoised results, run
```
python test_real_denoising_dnd.py 
```

- To reproduce PSNR/SSIM scores, upload the results to the DND benchmark [website](https://noise.visinf.tu-darmstadt.de/).
