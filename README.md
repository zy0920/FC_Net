# FC_Net

[GitHub](https://github.com/zy0920/FC_Net)

This project provides an implementation for "[基于DeFCN的贝类目标检测方法研究]" on PyTorch.

Experiments in the paper were conducted on the internal framework, thus we reimplement them on [cvpods](https://github.com/Megvii-BaseDetection/cvpods) and report details as below.

## Requirements
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
@article{wang2020end,
  title   =  {基于DeFCN的贝类目标检测方法研究},
  author  =  {Zhang, Yang},
  journal =  {},
  year    =  {2022}
}
```

## Contributing to the project
Any pull requests or issues about the implementation are welcome. If you have any issue about the library (e.g. installation, environments), please refer to [cvpods](https://github.com/Megvii-BaseDetection/cvpods).
