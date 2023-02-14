# 视觉-文本实体链接使用说明

## 1. 环境依赖

CUDA版本: 11.7
其他依赖库的安装命令如下：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 2. 下载安装

可使用如下命令下载安装算法包：
```bash
pip install -U mmkg-mm-entity-linking
```

## 3. 使用示例及运行参数说明

```python
from PIL import Image
from mmkg_mm_entity_linking import MMEntityLinking

image = Image.open("path/to/image")
probs = MMEntityLinking().inference(image, ["a dog", "a cat"])
```
