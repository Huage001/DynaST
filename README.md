# DynaST

This is the pytorch implementation of the following ECCV 2022 paper:

DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation

*Songhua Liu, Jingwen Ye, Sucheng Ren, and Xinchao Wang.*

<img src="https://github.com/Huage001/DynaST/blob/main/teaser.jpg" width="500px"/>

## Installation

```bash
git clone https://github.com/Huage001/DynaST.git
cd DynaST
conda create -n DynaST python=3.6
conda activate DynaST
pip install -r requirements.txt
```

## Inference

1. Prepare *DeepFashion* dataset following the instruction of [CoCosNet](https://github.com/microsoft/CoCosNet).

2. Create a directory for checkpoints if there is not:

   ```bash
   mkdir -p checkpoints/deepfashion/
   ```

3. Download pre-trained model from [here](https://drive.google.com/file/d/1UJ9xsQBBWZEXOz-jizerR4Qjo7n1kYmy/view?usp=sharing) and move the file to the directory '*checkpoints/deepfashion/*'.

4. Edit the file '*test_deepfashion.sh*' and set the argument '*dataroot*' to the root of the *DeepFashion* dataset.

5. Run:

   ```bash
   bash test_deepfashion.sh
   ```

6. Check the results in the directory '*checkpoints/deepfashion/test/*'.

## Training

1. Create a directory for the pre-trained VGG model if there is not:

   ```bash
   mkdir vgg
   ```

2. Download pre-trained VGG model used for loss computation from [here](https://drive.google.com/file/d/1xc7CBEsn45a3Pc3K9Tfh5wxKOx2LyC41/view?usp=sharing) and move the file to the directory '*vgg*'.

3. Edit the file '*train_deepfashion.sh*' and set the argument '*dataroot*' to the root of the *DeepFashion* dataset.

4. Run:

   ```bash
   bash train_deepfashion.sh
   ```

5. Checkpoints and intermediate results are saved in the directory '*checkpoints/deepfashion/*'.

## Citation

If you find this project useful in your research, please consider cite:

```
@Article{liu2022dynast,
    author  = {Songhua Liu, Jingwen Ye, Sucheng Ren, Xinchao Wang},
    title   = {DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation},
    journal = {European Conference on Computer Vision},
    year    = {2022},
}
```

## Acknowledgement

This code borrows heavily from [CoCosNet](https://github.com/microsoft/CoCosNet). We also thank the implementation of [Synchronized Batch Normalization](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).
