# Look Into Person

Human Parsing with DeepLabv3 in PyTorch.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

![image](https://github.com/foamliu/Look-Into-Person/raw/master/images/dataset.png)

Follow the [instruction](http://sysu-hcp.net/lip/index.php) to download Look-Into-Person dataset.

## Architecture

![image](https://github.com/foamliu/Look-Into-Person/raw/master/images/segnet.png)


## Usage
### Data Pre-processing
Extract training images:
```bash
$ python extract.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo

Download [pre-trained model](https://github.com/foamliu/Look-Into-Person/releases/download/v1.0/model.11-0.8409.hdf5) and put it into models folder.

```bash
$ python demo.py
```

Input | Merged | Output |
|---|---|---|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/0_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/0_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/0_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/1_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/1_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/2_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/2_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/2_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/3_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/3_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/3_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/4_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/4_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/4_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/5_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/5_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/6_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/6_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/6_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/7_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/7_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/7_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/8_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/8_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/8_out.png)|
|![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/9_image.png) | ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/9_merged.png)| ![image](https://github.com/foamliu/Look-Into-Person-v2/raw/master/images/9_out.png)|
