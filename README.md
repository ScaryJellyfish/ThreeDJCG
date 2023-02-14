# 3D Visual Captioning使用说明

## 1. 算法描述

3D Visual Captioning是一种以自动的方式为给定的3D视觉输入生成语法和语义上适当的描述的任务。为视觉输入生成解释性和相关的字幕不仅需要丰富的语言知识，还需要对视觉输入中出现的实体、场景及其交互有连贯的理解。

## 2. 环境依赖

CUDA版本: 11.3
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

其他依赖库的安装命令如下：
```bash
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## 3. 下载安装

可使用如下命令下载安装算法包：
```bash
pip install -U ThreeDJCG
```

## 4. 使用示例

### 输入：
data_dict: dict
        {
            point_clouds,
            lang_num，
            lang_feat_list，
            lang_len_list，
            main_lang_feat_list，
            main_lang_len_list，
            first_obj_list，
            unk_list，
            unk，
            istrain
        }

point_clouds: Variable(torch.cuda.FloatTensor)
        (B, N, 3 + input_channels) tensor
        Point cloud to run predicts on
        Each point in the point-cloud MUST
        be formated as (x, y, z, features...)

### 随机数据测试：

```python
from ThreeDJCG.model import MM3DJCG
import torch

data_dict = {}
point_clouds = torch.ones([8, 10, 4]).cuda()
data_dict['point_clouds'] = point_clouds

data_dict["lang_num"] = 1
data_dict["lang_feat_list"] = torch.zeros([1, 1, 126, 300]).cuda()
data_dict["lang_len_list"] = torch.Tensor([10]).cuda()
data_dict["main_lang_feat_list"] = torch.zeros([1, 1, 126, 300]).cuda()
data_dict["main_lang_len_list"] = torch.Tensor([10]).cuda()
data_dict["first_obj_list"] = torch.Tensor([0]).cuda()
data_dict["unk_list"] = torch.zeros([1, 300]).cuda()
data_dict["unk"] = torch.zeros([1, 300]).cuda()
data_dict["istrain"] = torch.Tensor([0]).cuda()

data_dict = MM3DJCG().inference(data_dict=data_dict)
```

## 5. 参数说明

```
point_clouds            点云数据 B, N, 3+1+128(multiview)
lang_num，              句子数，默认1 
lang_feat_list，        glove映射后的句子 B, N, 单词数, 映射维度（300）
lang_len_list，         句子单词数
main_lang_feat_list，   glove映射后的重要的一句，只有一句话则默认本身 B, N, 单词数, 映射维度（300）
main_lang_len_list，    句子单词数
first_obj_list，        句子中第一个重要的单词
unk_list，              glove中未对应的单词列表
unk，                   glove中未对应的单词
istrain                 是否为训练模式，为否
```

## 6. 论文引用

本项目代码使用了CVPR 2022的"3DJCG: A Unified Framework for Joint Dense Captioning and Visual Grounding on 3D Point Clouds"，其提出了一个联合的框架协同地解决3d dense captiong和3d visual grounding这两个紧密联合的任务，并取得了SOTA效果。