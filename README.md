# Traffic Collision Risk Forecasting Dashboard

基于 Attention-LSTM 的交通碰撞风险预测与可解释分析项目。项目包含 NYC 和 Chicago 双城交通时序数据、事故标签、空间邻接关系、模型训练脚本、前端导出脚本和可视化大屏。

## 项目定位

本项目是一个交通风险预测与可视化原型系统，不是实时生产级交通平台。

核心目标：

- 使用交通时序数据和事故标签训练风险预测模型
- 对比多种基线模型并进行消融实验
- 将模型输出、真实标签校验和空间邻接关系导出到前端
- 通过 `dashboard_fixed.html` 展示风险热力图、TP / FP / FN 校验、模型指标和可解释分析

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `nyc/` | NYC 数据、标签和邻接关系 |
| `chicago/` | Chicago 数据、标签和邻接关系 |
| `weights/` | 模型权重文件，使用 Git LFS 管理 |
| `results/` | 前端 JSON、指标和推理结果 |
| `dashboard_fixed.html` | 当前主要前端大屏 |
| `train.py` | 模型训练入口 |
| `infer_and_export_frontend.py` | 推理并导出前端数据 |
| `export_frontend_predictions.py` | 生成前端预测 JSON |
| `export_frontend_metrics.py` | 生成前端指标 JSON |
| `export_frontend_topology.py` | 生成前端拓扑 JSON |
| `论文图表汇总.md` | 论文流程图和图表索引 |
| `论文表格汇总.md` | 论文可用表格 |
| `答辩问答准备.md` | 答辩常见问题和回答 |

## Git LFS 注意事项

本项目包含较大的 `.npy` 和 `.h5` 文件，已使用 Git LFS 管理。

不要直接使用 GitHub 的 `Download ZIP`，否则可能只下载到 LFS 指针文件，导致 Python 读取数据时报“数据损坏”。

推荐下载方式：

```powershell
git lfs install
git clone https://github.com/harper886/trafficcompare.git
cd trafficcompare
git lfs pull
```

如果已经 clone 过项目：

```powershell
git lfs install
git lfs pull
```

检查大文件是否完整：

```powershell
Get-Item nyc\data_nyc.npy,chicago\data_chicago.npy
```

正常大小约为：

- `nyc/data_nyc.npy`：390 MB
- `chicago/data_chicago.npy`：110 MB

## 环境安装

建议使用 Python 3.9。

```powershell
pip install -r requirement.txt
```

如果需要 GPU 训练，TensorFlow 2.10 在 Windows 上通常对应 CUDA 11.2 和 cuDNN 8.1。只运行前端展示不需要 GPU。

## 启动前端

必须通过本地 Web 服务访问前端，不建议直接双击 HTML 文件。

```powershell
python -m http.server 8000
```

浏览器访问：

```text
http://localhost:8000/dashboard_fixed.html
```

前端默认加载：

- `results/frontend_predictions_nyc.json`
- `results/frontend_predictions_chicago.json`
- `results/frontend_metrics.json`
- `results/frontend_topology.json`

如果这些文件加载失败，页面会使用内置演示数据作为兜底。

## 数据集统计

| 数据集 | 时间片数量 | 区域数量 | 特征维度 | 样本总数 | 正样本比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NYC | 13,128 | 64 | 116 | 840,192 | 22.53% |
| Chicago | 8,784 | 27 | 116 | 237,168 | 13.38% |

## 训练模型

NYC：

```powershell
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 1
```

Chicago：

```powershell
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 1 --streaming_postprocess 1
```

消融实验：

```powershell
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 0 --streaming_postprocess 1
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 0
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 0 --streaming_postprocess 0
```

基线模型示例：

```powershell
python train.py --gpus 0 --dataset nyc --model lstm
python train.py --gpus 0 --dataset nyc --model gru
python train.py --gpus 0 --dataset nyc --model mlp
```

## 推理并导出前端数据

```powershell
python infer_and_export_frontend.py --dataset nyc --weights weights/myplan_nyc.h5
python infer_and_export_frontend.py --dataset chicago --weights weights/myplan_chicago.h5
```

也可以单独导出前端预测 JSON：

```powershell
python export_frontend_predictions.py --dataset nyc --output results/frontend_predictions_nyc.json
python export_frontend_predictions.py --dataset chicago --output results/frontend_predictions_chicago.json
```

导出指标和拓扑：

```powershell
python export_frontend_metrics.py
python export_frontend_topology.py
```

## 当前实验结果摘要

| 数据集 | 模型 | AUC-PR | AUC-ROC | F1 | Accuracy | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NYC | Myplan | 0.6955 | 0.8786 | 0.6443 | 0.7865 | 0.8265 |
| Chicago | Myplan | 0.5617 | 0.8290 | 0.4730 | 0.7733 | 0.7055 |

完整表格见 `论文表格汇总.md`。

## 前端展示数据说明

前端展示的是从本地数据和模型结果导出的代表性样本：

| 城市 | 展示时间帧 | 区域数 | 区域-时间记录 | 带标签校验记录 |
| --- | ---: | ---: | ---: | ---: |
| NYC | 13 | 64 | 832 | 296 |
| Chicago | 13 | 27 | 351 | 83 |

这样设计是为了避免浏览器直接加载大型 `.npy` 文件，同时保留代表性的风险变化和标签校验结果。

## 论文与答辩材料

项目中已经整理了以下辅助材料：

- `论文图表汇总.md`
- `项目流程图.md`
- `论文表格汇总.md`
- `答辩问答准备.md`
- `assets/project-intro.png`
- `assets/frontend-real-screenshot.png`
- `assets/paper-model-structure.png`
- `assets/paper-dataset-construction.png`
- `assets/paper-experiment-comparison.png`
- `assets/paper-ablation-study.png`
- `assets/paper-deployment-workflow.png`

## 答辩表述建议

可以表述为：

> 本系统使用 NYC 和 Chicago 交通时序数据、事故标签、空间邻接关系和模型评估结果完成训练与分析。前端为了保证交互性能，没有直接加载全量 `.npy` 文件，而是通过导出脚本抽取代表性时间帧，并生成前端可读 JSON 文件进行可视化展示。

不要表述为：

- 实时线上交通系统
- 已经接入完整真实地图坐标和实时事故接口
- 前端所有文字和区域描述都来自真实数据库
