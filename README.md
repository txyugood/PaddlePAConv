# PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds（PAConv 基于Paddle复现）
## 1.简介
该论文介绍了位置自适应卷积（PAConv），一种用于三维点云处理的通用卷积运算。PAConv的关键是通过动态组合存储在权重库中的基本权重矩阵来构造卷积矩阵，其中这些权重矩阵的系数通过核心网从点位置自适应学习。通过这种方式，内核构建在数据驱动管理器中，使PAConv比二维卷积具有更大的灵活性，可以更好地处理不规则和无序的点云数据。此外，通过组合权重矩阵而不是从点位置预测核，降低了学习过程的复杂性。

此外，与现有的点云卷积运算不同，他们的网络架构通常是经过精心设计的，我们将我们的PAConv集成到基于经典MLP的点云处理网络中，而不需要改变网络配置。即使建立在简单的网络上，我们的方法仍然接近甚至超过最先进的模型，并显著提高了分类和分割任务的基线性能并且效率相当高。


## 2.复现精度
在UCF-101数据集上spilt1的测试效果如下表。

| NetWork | epochs | opt | num points | batch_size | dataset | acc | vote acc | 
| --- | --- | --- | --- | --- | --- | --- | --- | 
| PAConv | 350 | SGD | 1024 | 32 | modelnet40 | 93.27% | 93.40% | 

## 3.数据集
modelnet40数据集下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/161759](https://aistudio.baidu.com/aistudio/datasetdetail/161759)

最优模型下载地址:

[https://pan.baidu.com/s/16h7UcMDzkCbkCP09slRaSw](https://pan.baidu.com/s/16h7UcMDzkCbkCP09slRaSw)

提取码: 2jh9 





## 4.环境依赖
PaddlePaddle == 2.3.1
## 5.快速开始
### 训练：

下载数据集解压后，将数据集放到项目的data目录下。

```shell
cd PaddlePAConv
nohup python -u main.py --config config/dgcnn_paconv_train.yaml --dataset_root data/modelnet40_ply_hdf5_2048 > train.log &
tail -f train.log
```
config: 配置文件地址

dataset_root: 训练集路径


### 测试：

使用最优模型进行评估.

```shell
 python -u main.py --config config/dgcnn_paconv_test.yaml --dataset_root data/modelnet40_ply_hdf5_2048 --model_path best_model.pdparams
```

config: 配置文件路径

dataset_root: 训练集路径

model_path: 预训练模型路径

测试结果

```shell
Loading pretrained model from best_model.pdparams
There are 85/85 variables loaded into PAConv.
Test :: test acc: 0.932739, test avg acc: 0.893860
```

投票测试使用以下命令:

```shell
 python -u eval_voting.py --config config/dgcnn_paconv_test.yaml --dataset_root data/modelnet40_ply_hdf5_2048 --model_path best_model.pdparams
```

测试结果

```shell
Voting 297, test acc: 93.152350,
Voting 298, test acc: 92.990276,
Voting 299, test acc: 93.233387,
Final voting test acc: 93.395462,
```

### 单张图片预测

输入的点云形状如下图：

<img src=./example/pred.png width=512></img>



```
python predict.py --config config/dgcnn_paconv_test.yaml --input_file example/plane.h5 --model_path best_model.pdparams
```
参数说明: 

config: 配置文件路径

input_file: 输入文件

model_path: 训练好的模型


```
Compiling user custom op, it will cost a few seconds.....
W0805 16:47:21.422516  5542 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0805 16:47:21.426019  5542 gpu_resources.cc:91] device: 0, cuDNN Version: 7.6.
Loading pretrained model from best_model.pdparams
There are 85/85 variables loaded into PAConv.
The input points is class airplane
```

### 模型导出
模型导出可执行以下命令：

```shell
python export_model.py --model_path best_model.pdparams --save_dir ./output/
```

参数说明：

model_path: 模型路径

save_dir: 输出图片保存路径

### Inference推理

可使用以下命令进行模型推理。该脚本依赖auto_log, 请参考下面TIPC部分先安装auto_log。infer命令运行如下：

```shell
python infer.py --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_file=output/model.pdmodel --batch_size=1 --input_file=example/plane.h5 --enable_benchmark=False --precision=fp32 --params_file=output/model.pdiparams
```

参数说明:

use_gpu:是否使用GPU

enable_mkldnn:是否使用mkldnn

cpu_threads: cpu线程数
 
model_file: 模型路径

batch_size: 批次大小

input_file: 输入文件路径

enable_benchmark: 是否开启benchmark

precision: 运算精度

params_file: 模型权重文件，由export_model.py脚本导出。


### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://gitee.com/Double_V/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/paconv/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/paconv/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

<img src=./test_tipc/data/tipc_result.png></img>


## 6.代码结构与详细说明
```shell
PaddlePAConv
├── README.md #用户指南
├── assign_score_gpu #自定义算子，运行时会自动编译加载
│   ├── __init__.py
│   ├── assign_score_withk_cuda.cc
│   ├── assign_score_withk_kernel.cu
│   └── setup.py
├── config #配置文件
│   ├── dgcnn_paconv_test.yaml # 训练脚本
│   └── dgcnn_paconv_train.yaml #测试脚本
├── data
│   └── modelnet40_ply_hdf5_2048 #数据集
├── eval_voting.py # 评估脚本 （投票）
├── example # 样例图片文件夹
├── export_model.py # 模型导出脚本
├── infer.py # 推理脚本
├── main.py # 主程序，包含训练和测试
├── model
│   ├── DGCNN_PAConv.py # 模型网络
│   ├── __init__.py
│   └── param_init.py
├── precise_bn.py 
├── test.log # 测试日志
├── test_tipc # TIPC测试目录
│   ├── README.md
│   ├── common_func.sh
│   ├── configs
│   ├── data
│   ├── docs
│   ├── output
│   ├── prepare.sh
│   └── test_train_inference_python.sh
├── train.log # 训练日志
├── util # 工具类
│   ├── PAConv_util.py
│   ├── __init__.py
│   ├── data_util.py
│   └── util.py
└── vis.py # 点云可视化脚本
```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| PAConv |
|框架版本| PaddlePaddle==2.3.1|
|应用场景| 点云分类 |
