# FC_Net

[GitHub](https://github.com/zy0920/FC_Net)

This project provides an implementation for "FC_Net" on PyTorch.

Experiments in the paper were conducted on the internal framework, thus we reimplement them on [cvpods](https://github.com/Megvii-BaseDetection/cvpods) and report details as below.

＃＃ 原创数据集
![%2_$ U{WUA CVP(}H`9G%0R](https://user-images.githubusercontent.com/102900203/166199452-1dcbd0e6-a268-42c7-9c9c-e8450b2f6a88.png)
![图5](https://user-images.githubusercontent.com/102900203/166203478-900a4f2e-e92e-4f35-a517-0ee12d7f0528.png)
![图6](https://user-images.githubusercontent.com/102900203/166203493-2a7c4956-4ee8-4a50-ac97-c6bdce079932.png)
![图7](https://user-images.githubusercontent.com/102900203/166203446-cfabc365-641c-477e-90f1-d37dee058cdc.png)

＃＃ 实验结果
![图10](https://user-images.githubusercontent.com/102900203/166203568-b36ea005-a0a5-4f18-9bbb-b549be5fc3e1.png)
![图11](https://user-images.githubusercontent.com/102900203/166203579-1b8c8c98-d911-45b3-99f2-d77d7396a79d.png)

＃＃ 技术路线图
![图1](https://user-images.githubusercontent.com/102900203/166203671-9b62124a-5b25-4d21-8f2a-a69b28a4c43f.png)

＃＃ 要求
* [cvpods](https://github.com/Megvii-BaseDetection/cvpods)
* scipy >= 1.5.4

## Get Started

* install cvpods locally (requires cuda to compile)
```shell

python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

# Or,
pip install -r requirements.txt
python3 setup.py build develop
```

* prepare datasets
```shell
cd /path/to/cvpods
cd datasets
ln -s /path/to/your/dataset
```

* Train & Test
```shell
git clone https://github.com/zy0920/FC_Net.git
cd DeFCN/playground/detection/coco/poto.res101.fpn.coco.800size.3x_ms0403  # for example

# Train
pods_train --num-gpus 8

# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"

```

## Acknowledgement
This repo is developed based on DeFCN. Please check [DeFCN](https://github.com/Megvii-BaseDetection/DeFCN) for more details and features.

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

## Citing
If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@article{Zhangyang2022end,
  title   =  {FC_Net from Zhangyang},
  author  =  {Zhang, Yang},
  journal =  {},
  year    =  {2022}
}
```

## Contributing to the project
Any pull requests or issues about the implementation are welcome. If you have any issue about the library (e.g. installation, environments), please refer to [cvpods](https://github.com/Megvii-BaseDetection/cvpods).
