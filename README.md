# Train model 

```bash
# NYC
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 1
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 0 --streaming_postprocess 1
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 0
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 0 --streaming_postprocess 0

# Chicago
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 1 --streaming_postprocess 1
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 0 --streaming_postprocess 1
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 1 --streaming_postprocess 0
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 0 --streaming_postprocess 0

# Baselines
python train.py --gpus 0 --dataset nyc --model lstm
python train.py --gpus 0 --dataset nyc --model gru
python train.py --gpus 0 --dataset nyc --model mlp
```

如果想本地保存模型结果

```bash
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 1 --save_weights weights/myplan_nyc.h5
```

# 导出前端可视化预测结果

如果你已经有模型输出的逐窗逐网格概率（例如 `pred.npy` / `smooth.npy`），可以将其导出为前端可直接读取的 JSON：

```bash
# NYC 示例
python export_frontend_predictions.py --dataset nyc --pred results/nyc_pred.npy --smooth results/nyc_smooth.npy --threshold 0.31 --output results/frontend_predictions_nyc.json

# Chicago 示例
python export_frontend_predictions.py --dataset chicago --pred results/chicago_pred.npy --smooth results/chicago_smooth.npy --threshold 0.31 --output results/frontend_predictions_chicago.json
```

说明：

- `pred.npy` 应为模型原始概率输出，长度 = `窗口数 × 区域数`
- `smooth.npy` 应为流式后处理 / 平滑后的概率输出，长度同上
- 若未提供 `smooth.npy`，则前端将直接使用 `pred.npy`
- 若两者都未提供，则该脚本只能导出真实标签与真实网格结构，不能形成真实预测热力图

当前 `dashboard.html` 若要进一步升级为“更真实版”，建议下一步改为优先读取：

- `results/frontend_predictions_nyc.json`
- `results/frontend_predictions_chicago.json`

这样地图热力、TP / FN / FP、时序演化都可以直接基于真实模型输出驱动。

# 从权重直接推理并导出前端 JSON

如果你已经有训练好的权重，可以直接一步生成前端使用的真实预测文件：

```bash
# NYC
python infer_and_export_frontend.py --dataset nyc --weights weights/myplan_nyc.h5

# Chicago
python infer_and_export_frontend.py --dataset chicago --weights weights/myplan_chicago.h5
```

默认会输出：

- `results/nyc_pred.npy`
- `results/nyc_smooth.npy`
- `results/frontend_predictions_nyc.json`
- `results/chicago_pred.npy`
- `results/chicago_smooth.npy`
- `results/frontend_predictions_chicago.json`

可选参数：

```bash
python infer_and_export_frontend.py \
  --dataset nyc \
  --weights weights/myplan_nyc.h5 \
  --threshold 0.31 \
  --max_neigh 4
```

说明：

- 若未显式指定 `--threshold`，脚本会尝试从 `results/metrics.jsonl` 中读取当前数据集最近一次 `myplan` 的阈值
- 若加上 `--no_streaming`，则不会做流式后处理，前端将直接使用原始预测概率
- 该脚本依赖 `Python 3.9 + TensorFlow 2.10.0`，需与你当前训练环境一致

# Requirements

- Python 3.9
- TensorFlow 2.10.0
- 或 TensorFlow 2.10.0（GPU版本需要 CUDA 11.2 + cuDNN 8.1，Windows GPU 最后一个官方支持版本）
- 其他依赖见 `requirement.txt`

安装方式：

```bash
pip install -r requirement.txt

查看数据集文件：
python test.py nyc/data_nyc.npy --max_print 20
python test.py nyc/data_nyc.npy --stats

查看标签文件：
python test.py nyc/label.npy --max_print 20
python test.py nyc/label.npy --stats

```
