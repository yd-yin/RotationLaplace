# [ICLR2023 Spotlight] A Laplace-inspired Distribution on SO(3) for Probabilistic Rotation Estimation

Project page: https://pku-epic.github.io/RotationLaplace/

# Setup

## Dependencies

- PyTorch=1.10.1 + torchvision=0.11.2 + cudatoolkit=11.3

- PyTorch3D (Installation via `conda` or `pypi` may be problematic. Build from source if necessary.)
  ```bash
  git clone https://github.com/facebookresearch/pytorch3d.git
  PYTORCH3D_FORCE_NO_CUDA=1 pip install -e pytorch3d    # consider using CPU only version if there are issues with cuda installation
  ```

- Other dependencies

  ```bash
  pip install opencv-python tqdm matplotlib scipy lmdb pyyaml wget scikit-image tensorboard tensorboardX
  ```

## Dataset

**ModelNet10-SO(3)**

Obtain ModelNet10-SO(3) dataset from [website](https://github.com/leoshine/Spherical_Regression#modelnet10-so3-dataset) and link it to `datasets`

```bash
unzip ModelNet10-SO3.zip
ln -s $PWD/ModelNet10-SO3 $PROJECT_PATH/datasets
```

**Pascal3D+**

Obtain Pascal3D+ (release1.1) dataset from [website](https://cvgl.stanford.edu/projects/pascal3d.html) and link it to `datasets`

```bash
unzip PASCAL3D+_release1.1.zip
ln -s $PWD/PASCAL3D+_release1.1 $PROJECT_PATH/datasets
```

Obtain the synthetic data from [website](https://shapenet.cs.stanford.edu/media/syn_images_cropped_bkg_overlaid.tar) and link it to `datasets`
```bash
tar -xvf syn_images_cropped_bkg_overlaid.tar
ln -s $PWD/syn_images_cropped_bkg_overlaid $PROJECT_PATH/datasets
```

Please note that when using Pascal3D+, the data annotations will be generated during the first run of the program.


## Equivolumetric distretization of SO(3)
We used the code from Implicit-PDF (https://implicit-pdf.github.io/) to generate equivolumetric samples of SO(3).

By default, we used `grids3.npy`, which is stored in this repository.
For more pre-sampled grids, please refer to this [Google drive](https://drive.google.com/drive/folders/188-ivOJleXXQZPUjWIBxwkW5RrZIhyUG?usp=sharing).
The Eq_grids files contain different numbers of samples for both the rotation matrix (grids{l}) and the quaternion (gridsq{l}). 
The number of samples is 72 * 8^l.


# Usage

## Train

```bash
python main.py <exp_name> <configs> [--args]
```
The training process is logged by `tensorboard`.

### ModelNet10-SO3:
```bash
python main.py RLaplace configs/modelnet_config.json --loss=RLaplace --eval_freq=2000 -g=0 
python main.py RLaplace_sofa configs/modelnet_config.json --loss=RLaplace --category=sofa --eval_freq=200 -g=0 
```

### Pascal3D+
There are two different experimental settings related to the evaluation data source.

In Sec 5.2, we follow [IPDF](https://arxiv.org/abs/2106.05965) to use (the more
challenging) PascalVOC val as the test set.
```bash
python main.py RLaplace_pascal configs/pascal3d_synth_both_pascal.json --loss=RLaplace --eval_freq=2000 -g=0
```

In Sec 5.3, we follow [RPMG](https://arxiv.org/abs/2110.11657) to use ImageNet_val as the test set.
```bash
python main.py RLaplace_imagenet_sofa configs/pascal3d_synth.json --loss=RLaplace --category=sofa --eval_freq=200 -g=0
```

## Evaluation

```bash
python main.py <exp_name> <configs> --eval_only [--args]
```
To specify the trained model, set the `CKPT` environment variable.
If no specific checkpoint is specified, the model will default to using the checkpoint from the final epoch of training.

```bash
CKPT=release python main.py RLaplace configs/modelnet_config.json --loss=RLaplace --eval_only -g=0
CKPT=release python main.py RLaplace_pascal configs/pascal3d_synth_both_pascal.json --loss=RLaplace --eval_only -g=0
```

# Bibtex

```bibtex
@InProceedings{yin2022fishermatch,
  author={Yin, Yingda and Wang, Yang and Wang, He and Chen, Baoquan},
  title={A Laplace-inspired Distribution on SO(3) for Probabilistic Rotation Estimation},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
}
```

# Acknowledgement
The code base used in this project is sourced from the repository of the matrix Fisher distribution, https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions.

For a fair comparison, we have primarily adhered to the original implementation details. Besides, we have introduced support for different test data sources of Pascal3D+ dataset.