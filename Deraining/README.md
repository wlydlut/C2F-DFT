## Datasets

-Train datasets: [Rain13K](https://drive.google.com/file/d/14BidJeG4nSNuFNFDf99K-7eErCq4i47t/view?usp=sharing)

-Test datasets: [Rain100L, Rain100H, Test100, Test2800](https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view?usp=sharing)

-Val dataset: To accelerate the training speed, we selected the first image from the Rain100L test set as the validation set.

- The above dataset path is as follows
  
 `Deraining/Datasets` <br/>
 `├──train`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
 `├──val`  <br/>
          `├──input`   <br/>
          `└──gt`   <br/>
 `└──test`  <br/>
     `├──Test100`   <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `├──Rain100H`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `├──Rain100L`  <br/>
          `├──input`   <br/>
          `└──target`   <br/>
     `└──Test2800`<br/>
          `├──input`   <br/>
          `└──target` 
## Training

1. To train URDT in the coarse training pipeline, modify the comments on lines 129-134 and 195-237 in the /basicsr/models/image_restoration_model.py file, then run

    ```
    cd URDT-main
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt Deraining/Options/Deraining_URDT_Coarse.yml  --launcher pytorch
    ```

2. To train URDT in the fine training pipeline, modify the comments on lines 137-145 and 240-286 in the /basicsr/models/image_restoration_model.py file, then run

    ```
    cd URDT-main
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt Deraining/Options/Deraining_URDT_Fine.yml  --launcher pytorch
    ```

## Testing

1. Download the pre-trained [model](https://drive.google.com/drive/folders/18dVkwv9WUBXMaCLtsuzA4TYURDtW_DxG) and place it in `./pretrained_models/`

2. Testing
    ```
    cd Deraining
    python test.py
    ```

3. Calculating PSNR/SSIM scores, run

    ```
    python calculate_psnr_ssim.py
    ```
