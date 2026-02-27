# 项目结构

项目根目录/
│
├── README.md                                   # 项目说明文档 
│
├──基础推理测试                      # 简单的对单张图片推理
│ 
├──labels integration.py              # 将COCO数据集提供的标签融合为标准coco格式
│
├──coco2odvg(Open-GroundingDINO提供).py #将coco格式标签转换为odvg格式
│
├──eval on coco.py                             # 验证groundingdino在COCO验证集上的表现
│
├──prompt_comparison.py       # 提示词工程，比较三种提示词
│
├──random sampling.py        # 将标签文件缩小(随机采样),减少训练的样本量
│
├──split_odvg_seen65unseen15.py      # 分割数据集(65seen和15unseen) 
│
├──val labels split seen unseen to coco.py  # 验证集分割并转为COCO格式(不变)
│
├──trained_model_eval.py              # 验证训练后的模型在unseen类上的表现
│
├──visualized prompt comparison.py   # 三种提示词的可视化对比
│ 
├──Prompt Ensembling.py       # 多提示词集成
│
├──train.bat   #在open-groundingdino环境下运行，<u>其中的路径需做对应修改</u>
│
└──训练配置文件   # 训练groundingdino的配置文件，<u>注意数据集和配置文件路径</u>

***<u>运行时请对应修改代码中的权重和模型配置文件路径以及数据集路径并根据代码中存放结果的路径在项目中创建对应的目录。推理时需要bert-base-uncased，可尝试手动下载，并放在程序的同级目录下。</u>***

# 环境配置

### 1.GroundingDINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
conda create -n your_env python=3.8
conda activate your_env
cd /d paht/to/groundingdino-main
pip install -e .
#下载预训练权重
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

### 2.Open-GroundingDINO

```bash
git clone https://github.com/longzw1997/Open-GroundingDino.git && cd Open-GroundingDino/      
conda create -n your_env python=3.7.11
conda activate your_env
cd /d path/to/open-groundingdino-main
pip install -r requirements.txt
cd /d path/to/open-groundingdino-main/models/GroundingDINO/ops
python setup.py build install
```

> [!NOTE]
>
> 1. 安装open-groundingdino的依赖时，如果安装safetensors(transformers的依赖)时报错，尝试手动安装低版本的safetensors
> 2. 运行setup.py时需要C++环境，在C++环境下运行anaconda的activate.bat，进入base环境后再激活相关环境并运行`python setup.py build install`

# 训练

安装完依赖后，将训练配置文件夹中的文件放在.../open-groundingdino-main/config下，***<u>根据数据集的位置与训练需求做对应的修改</u>***，最后在终端中运行train.bat

