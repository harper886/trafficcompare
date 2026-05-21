# 毕业答辩 PPT 项目详细说明

## 1. 项目基本信息

**项目名称：** 基于Attention-LSTM的交通碰撞异常检测预警研究与分析

**项目类型：** 交通安全预测、深度学习、时空数据分析、前端可视化系统

**项目定位：** 本项目是一个基于离线数据的交通碰撞风险预测与可视化原型系统。系统使用 NYC 和 Chicago 两个城市的交通时序数据、事故标签和空间邻接关系进行建模实验，并将模型结果导出到前端大屏中展示。

答辩时需要特别说明：本项目不是实时线上交通平台，也没有接入实时地图服务或实时事故接口。前端展示的是由本地数据和模型结果导出的代表性样本，用于展示模型效果和系统功能。

## 2. 项目研究背景

城市交通碰撞事故具有突发性、区域性和时间波动性。某些区域在早晚高峰、节假日、特殊道路结构或交通流量变化时更容易出现风险。如果能够提前识别高风险区域和高风险时间段，就可以为交通管理、道路安全预警和应急调度提供辅助依据。

传统统计方法通常难以同时处理交通数据中的时间依赖和空间关联。例如，某一区域的风险不仅和当前时刻的交通状态有关，也可能受到过去多个时间片的影响；同时，相邻区域之间也可能存在风险传播或相似变化。因此，本项目选择使用 Attention-LSTM 模型对交通碰撞风险进行预测，并结合前端可视化系统对结果进行展示。

PPT 中这一部分可以突出三个问题：

- 交通碰撞风险具有明显的时序变化。
- 不同区域之间存在空间关联。
- 类别不平衡场景下，发现高风险样本比单纯追求准确率更重要。

## 3. 项目研究目标

本项目的主要目标包括：

1. 构建 NYC 和 Chicago 双城交通碰撞风险预测数据流程。
2. 使用交通时序特征、事故标签和空间邻接关系训练风险预测模型。
3. 设计 Attention-LSTM 模型，提升对关键时间片和风险变化的表达能力。
4. 通过多模型对比实验和消融实验验证模型效果。
5. 将预测结果、模型指标和空间拓扑关系导出为前端 JSON。
6. 实现可视化大屏，展示风险热力图、预测校验、指标对比和可解释分析。

答辩时可以概括为：

> 本项目完成了从数据处理、模型训练、实验评估到前端可视化展示的完整流程，目标是对城市交通碰撞风险进行离线预测和可解释展示。

## 4. 数据集说明

项目使用两个城市数据集：NYC 和 Chicago。

| 数据集 | 时间片数量 | 区域数量 | 特征维度 | 样本总数 | 正样本数 | 负样本数 | 正样本比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NYC | 13,128 | 64 | 116 | 840,192 | 189,256 | 650,936 | 22.53% |
| Chicago | 8,784 | 27 | 116 | 237,168 | 31,733 | 205,435 | 13.38% |

主要数据文件包括：

- `E:\VScodeProject\trafficcompare\nyc\data_nyc.npy`：NYC 交通时序特征数据。
- `E:\VScodeProject\trafficcompare\nyc\label.npy`：NYC 事故标签。
- `E:\VScodeProject\trafficcompare\nyc\dict_xy.npy`：NYC 区域坐标映射。
- `E:\VScodeProject\trafficcompare\chicago\data_chicago.npy`：Chicago 交通时序特征数据。
- `E:\VScodeProject\trafficcompare\chicago\label.npy`：Chicago 事故标签。
- `E:\VScodeProject\trafficcompare\chicago\dict_xy.npy`：Chicago 区域坐标映射。
- `E:\VScodeProject\trafficcompare\nyc\road_ad.txt / E:\VScodeProject\trafficcompare\chicago\road_ad.txt`、`E:\VScodeProject\trafficcompare\nyc\poi_ad.txt / E:\VScodeProject\trafficcompare\chicago\poi_ad.txt`、`E:\VScodeProject\trafficcompare\nyc\record_ad.txt / E:\VScodeProject\trafficcompare\chicago\record_ad.txt`：道路、POI、历史记录相关邻接关系。

PPT 中不要把所有文件都列出来，可以只放数据统计表和数据来源说明。讲解时强调：每个样本可以理解为“某个区域在某个时间片的交通状态和风险标签”。

## 5. 项目整体流程

项目整体流程可以分为五个阶段：

1. **数据准备：** 整理 NYC 和 Chicago 的交通时序数据、事故标签、区域坐标和邻接关系。
2. **模型训练：** 使用 Attention-LSTM 进行风险预测建模。
3. **实验评估：** 与 LSTM、GRU、MLP、XGBoost、LightGBM 等模型进行对比，并进行消融实验。
4. **结果导出：** 将模型预测结果、评估指标和拓扑关系导出为前端可读取的 JSON 文件。
5. **前端展示：** 通过 `E:\VScodeProject\trafficcompare\dashboard_fixed.html` 展示风险热力图、时间轴、预测校验和模型指标。

推荐 PPT 图示素材：

- `E:\VScodeProject\trafficcompare\assets\chapter3_overall_project_diagram.png`
- `E:\VScodeProject\trafficcompare\assets\project-intro.png`
- `E:\VScodeProject\trafficcompare\assets\paper-deployment-workflow.png`

## 6. 模型方法说明

本项目核心模型为 Attention-LSTM。

LSTM 适合处理时间序列数据，可以学习历史交通状态对当前风险的影响。交通碰撞风险并不是只由当前时刻决定的，前几个时间片的交通变化、拥堵状态或异常波动都可能影响后续风险。

Attention 机制用于突出关键时间片或关键特征。不是所有历史时间片对预测结果都有相同贡献，注意力机制可以让模型更加关注对风险判断更重要的信息。

模型整体可以理解为：

```text
历史交通特征序列
  -> LSTM 时序编码
  -> Attention 权重分配
  -> 风险概率输出
  -> 阈值判断与后处理
  -> 高风险/低风险结果
```

推荐 PPT 图示素材：

- `E:\VScodeProject\trafficcompare\assets\paper-model-structure.png`
- `E:\VScodeProject\trafficcompare\assets\model_overall_architecture_flow.png`
- `E:\VScodeProject\trafficcompare\assets\attention_lstm_fusion_diagram.png`

答辩时可以这样解释：

> LSTM 负责学习交通数据随时间变化的规律，Attention 机制负责突出关键时间片和关键特征。两者结合后，模型可以更好地识别交通碰撞高风险样本。

## 7. 平滑与后处理机制

交通风险预测结果如果在相邻时间片之间频繁跳变，会影响预警系统的稳定性。为了解决这个问题，项目中加入了平滑门和流式后处理机制。

平滑门主要作用于模型内部动态特征，使风险表示更加稳定。流式后处理主要作用于模型输出端，减少预测概率在阈值附近频繁波动导致的误报或漏报。

推荐 PPT 图示素材：

- `E:\VScodeProject\trafficcompare\assets\hysteresis_threshold_postprocess_diagram.png`
- `E:\VScodeProject\trafficcompare\assets\paper-ablation-study.png`

答辩表述建议：

> 平滑门和 Streaming 后处理并不是替代模型主体，而是对风险变化和输出结果进行稳定化处理。消融实验表明，完整模型在两个城市数据集上的 Recall 表现更好。

## 8. 实验设计

项目实验主要包括两类：

1. **模型对比实验：** 将 Myplan 与传统机器学习模型、时序模型和已有交通预测相关模型进行比较。
2. **消融实验：** 去掉平滑门、去掉流式后处理，观察模型性能变化。

评价指标包括：

- **AUC-PR：** 更适合类别不平衡任务，反映模型识别正样本的能力。
- **AUC-ROC：** 反映模型整体排序能力。
- **F1：** 综合考虑 Precision 和 Recall。
- **Accuracy：** 整体分类准确率。
- **Recall：** 召回率，表示高风险样本被识别出来的比例。

交通碰撞预测中不要只强调 Accuracy。因为数据存在类别不平衡，如果模型倾向于预测多数类，Accuracy 可能看起来较高，但漏掉真正高风险样本。答辩时应重点强调 AUC-PR、F1 和 Recall。

## 9. 核心实验结果

主模型最终结果如下：

| 数据集 | 模型 | AUC-PR | AUC-ROC | F1 | Accuracy | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NYC | Myplan | 0.6955 | 0.8786 | 0.6443 | 0.7865 | 0.8265 |
| Chicago | Myplan | 0.5617 | 0.8290 | 0.4730 | 0.7733 | 0.7055 |

结果说明：

- NYC 数据集上，Myplan 的 AUC-PR、AUC-ROC、F1 和 Recall 表现较好。
- Chicago 数据集上，Myplan 也保持了较好的风险识别能力。
- Recall 较高说明模型能够识别出较多高风险样本，这对交通安全预警任务比较重要。

指标来源链路：

```text
E:\VScodeProject\trafficcompare\docs\paper\p1结果展示.md
  -> E:\VScodeProject\trafficcompare\export_frontend_metrics.py
  -> E:\VScodeProject\trafficcompare\results\frontend_metrics.json
  -> E:\VScodeProject\trafficcompare\dashboard_fixed.html
```

论文和答辩使用的整理版表格位于：

```text
E:\VScodeProject\trafficcompare\docs\paper\论文表格汇总.md
E:\VScodeProject\trafficcompare\docs\project\最终指标汇总与来源说明.md
```

## 10. 消融实验说明

消融实验用于说明模型中的平滑门和流式后处理是否有效。

| 配置 | NYC Recall | Chicago Recall |
| --- | ---: | ---: |
| 去平滑门 + 去流式后处理 | 0.7108 | 0.6318 |
| 去平滑门 | 0.8046 | 0.6693 |
| 去流式后处理 | 0.7407 | 0.6401 |
| 完整模型 | 0.8265 | 0.7055 |

可以看到，完整模型在 NYC 和 Chicago 上的 Recall 都高于去除模块后的版本。这说明平滑门和流式后处理对风险识别能力和输出稳定性有一定帮助。

答辩表述建议：

> 消融实验说明，自适应平滑门主要提升动态特征表达的稳定性，Streaming 后处理主要减少输出端的频繁抖动。两者配合后，模型对高风险样本的识别更稳定。

## 11. 前端可视化系统说明

项目实现了一个前端可视化大屏，主页面为：

```text
E:\VScodeProject\trafficcompare\dashboard_fixed.html
```

前端主要展示内容包括：

- 城市风险热力图。
- 时间轴播放和时间片切换。
- TP、FP、FN 预测校验。
- 模型指标对比。
- 区域风险概率和邻接关系解释。
- NYC 和 Chicago 双城切换。

前端默认读取的数据包括：

```text
E:\VScodeProject\trafficcompare\results\frontend_predictions_nyc.json
E:\VScodeProject\trafficcompare\results\frontend_predictions_chicago.json
E:\VScodeProject\trafficcompare\results\frontend_metrics.json
E:\VScodeProject\trafficcompare\results\frontend_topology.json
```

推荐 PPT 图示素材：

- `E:\VScodeProject\trafficcompare\assets\frontend-real-screenshot.png`
- `E:\VScodeProject\trafficcompare\assets\promo-dashboard-showcase.png`

答辩时需要说明：

> 前端不是直接加载全量 `.npy` 数据，而是读取导出的代表性 JSON 样本。这样做可以保证浏览器交互性能，也便于答辩现场稳定展示。

## 12. 项目运行说明

运行前端时，建议在项目根目录执行：

```powershell
python -m http.server 8000
```

然后在浏览器访问：

```text
http://localhost:8000/dashboard_fixed.html
```

不要直接双击 HTML 文件，因为前端需要通过 HTTP 服务加载 `E:\VScodeProject\trafficcompare\results\` 目录下的 JSON 文件。

如果 8000 端口被占用，可以使用：

```powershell
python -m http.server 8010
```

然后访问：

```text
http://localhost:8010/dashboard_fixed.html
```

## 13. GitHub 与部署说明

项目已经上传到 GitHub。由于项目包含较大的 `.npy` 和 `.h5` 文件，数据和模型权重使用 Git LFS 管理。

下载项目时建议使用：

```powershell
git lfs install
git clone https://github.com/harper886/trafficcompare.git
cd trafficcompare
git lfs pull
```

不建议直接使用 GitHub 的 Download ZIP，因为可能只下载到 Git LFS 指针文件，导致数据或模型权重无法正常读取。

## 14. PPT 推荐页数与内容

建议毕业答辩 PPT 控制在 12 到 14 页。

| 页码 | 标题 | 主要内容 | 推荐素材 |
| --- | --- | --- | --- |
| 1 | 封面 | 题目、姓名、专业、指导老师 | `E:\VScodeProject\trafficcompare\assets\promo-defense-cover.png` |
| 2 | 研究背景 | 交通碰撞风险预测意义 | `E:\VScodeProject\trafficcompare\assets\project-intro.png` |
| 3 | 研究目标 | 项目要解决的问题 | 流程要点 |
| 4 | 数据集说明 | NYC、Chicago 数据统计 | 数据统计表 |
| 5 | 总体流程 | 数据、训练、评估、导出、展示 | `E:\VScodeProject\trafficcompare\assets\chapter3_overall_project_diagram.png` |
| 6 | 模型结构 | Attention-LSTM 结构 | `E:\VScodeProject\trafficcompare\assets\paper-model-structure.png` |
| 7 | 后处理机制 | 平滑门、Streaming 后处理 | `E:\VScodeProject\trafficcompare\assets\hysteresis_threshold_postprocess_diagram.png` |
| 8 | 实验设计 | 对比模型、评价指标 | `E:\VScodeProject\trafficcompare\assets\paper-experiment-comparison.png` |
| 9 | 实验结果 | 核心指标表 | 主模型结果表 |
| 10 | 消融实验 | 模块有效性验证 | `E:\VScodeProject\trafficcompare\assets\paper-ablation-study.png` |
| 11 | 系统展示 | 前端大屏功能 | `E:\VScodeProject\trafficcompare\assets\frontend-real-screenshot.png` |
| 12 | 运行与部署 | 本地服务、Git LFS | `E:\VScodeProject\trafficcompare\assets\paper-deployment-workflow.png` |
| 13 | 总结 | 工作量和成果 | 条目总结 |
| 14 | 不足与展望 | 离线系统、后续改进 | 简洁列表 |

## 15. 每页讲稿示例

### 第 1 页：封面

各位老师好，我的毕业设计题目是《基于 Attention-LSTM 的交通碰撞风险预测与可解释分析系统》。本项目围绕城市交通碰撞风险预测问题，使用 NYC 和 Chicago 双城交通数据进行建模实验，并实现了一个前端可视化大屏，用于展示风险预测、真实标签校验和可解释分析结果。

### 第 2 页：研究背景

城市交通碰撞事故具有明显的时间波动和空间关联。某些区域在特定时间段可能更容易出现事故风险。传统统计方法难以同时捕捉历史时间变化和区域之间的关联，因此本项目希望利用深度学习模型对交通碰撞风险进行预测，为交通安全预警提供辅助。

### 第 3 页：研究目标

本项目主要完成四个目标：第一，构建双城交通碰撞预测数据流程；第二，训练 Attention-LSTM 风险预测模型；第三，完成模型对比和消融实验；第四，将预测结果导出到前端大屏中进行可视化展示。

### 第 4 页：数据集说明

项目使用 NYC 和 Chicago 两个城市数据集。NYC 包含 13,128 个时间片、64 个区域，Chicago 包含 8,784 个时间片、27 个区域。每个样本由交通时序特征和事故标签组成，同时结合道路、POI 和历史记录等邻接关系描述区域之间的空间联系。

### 第 5 页：总体流程

系统流程从数据构建开始，先整理交通时序数据、事故标签和空间邻接关系；然后训练风险预测模型；接着通过对比实验和消融实验评估模型效果；最后把预测结果、指标和拓扑关系导出为 JSON，并在前端页面中展示。

### 第 6 页：模型结构

模型采用 Attention-LSTM 结构。LSTM 用于学习历史交通状态中的时间依赖，Attention 机制用于突出关键时间片和关键特征。模型最终输出每个区域在目标时间片上的风险概率，再通过阈值和后处理得到风险判断结果。

### 第 7 页：后处理机制

交通风险在连续时间片中通常具有一定连续性。如果预测结果频繁跳变，会影响预警展示的稳定性。因此项目加入平滑门和 Streaming 后处理机制，用于减少风险概率和预警结果在阈值附近的抖动。

### 第 8 页：实验设计

实验中将本文模型与多种传统机器学习模型、统计方法和深度学习模型进行对比。评价指标包括 AUC-PR、AUC-ROC、F1、Accuracy 和 Recall。由于交通碰撞样本存在类别不平衡，所以本文重点关注 AUC-PR、F1 和 Recall。

### 第 9 页：实验结果

从结果看，Myplan 在 NYC 数据集上取得了 0.6955 的 AUC-PR、0.8786 的 AUC-ROC 和 0.8265 的 Recall；在 Chicago 数据集上取得了 0.5617 的 AUC-PR、0.8290 的 AUC-ROC 和 0.7055 的 Recall。结果说明模型具备一定的高风险样本识别能力。

### 第 10 页：消融实验

消融实验中分别去除平滑门和 Streaming 后处理。完整模型在两个城市上的 Recall 都高于去除模块后的版本，说明这两个模块对提升风险识别能力和输出稳定性有帮助。

### 第 11 页：系统展示

前端页面支持 NYC 和 Chicago 城市切换，可以查看不同时间片下的风险热力图，并展示 TP、FP、FN 等预测校验结果。同时，页面还展示模型指标、区域风险概率和空间邻接关系，让预测结果更直观、可解释。

### 第 12 页：运行与部署

项目通过本地 HTTP 服务运行前端页面。运行时在项目根目录执行 `python -m http.server 8000`，然后访问 `E:\VScodeProject\trafficcompare\dashboard_fixed.html`。项目中的 `.npy` 数据和 `.h5` 权重使用 Git LFS 管理，因此下载项目后需要执行 `git lfs pull`。

### 第 13 页：总结

总体来说，本项目完成了交通碰撞风险预测的完整流程，包括数据构建、模型训练、实验评估、结果导出和前端可视化展示。实验结果说明 Attention-LSTM 结合平滑和后处理机制能够较好地识别交通碰撞高风险样本。

### 第 14 页：不足与展望

本项目目前仍是离线原型系统，尚未接入实时交通接口和真实 GIS 地图边界。后续可以继续引入天气、施工、大型活动等外部因素，也可以进一步扩展为在线预测和实时预警系统。

## 16. 答辩常见问题准备

### 问题 1：这个系统是实时的吗？

不是。当前系统是基于离线数据的预测与可视化原型。前端展示的是从本地数据和模型结果中导出的代表性样本，没有接入实时交通接口。

### 问题 2：为什么使用 Attention-LSTM？

LSTM 适合处理交通时序数据，能够学习历史状态对当前风险的影响。Attention 机制可以突出关键时间片和关键特征，使模型更关注对风险判断更重要的信息。

### 问题 3：为什么不只看 Accuracy？

交通碰撞预测是类别不平衡任务，正样本相对较少。如果只看 Accuracy，模型可能偏向多数类而漏掉高风险样本。因此本项目更关注 AUC-PR、F1 和 Recall。

### 问题 4：指标从哪里来的？

指标首先整理在 `E:\VScodeProject\trafficcompare\docs\paper\p1结果展示.md` 和 `E:\VScodeProject\trafficcompare\docs\paper\论文表格汇总.md` 中，然后通过 `E:\VScodeProject\trafficcompare\export_frontend_metrics.py` 导出为 `E:\VScodeProject\trafficcompare\results\frontend_metrics.json`，前端页面再读取该 JSON 文件展示。

### 问题 5：前端为什么不直接读取 `.npy`？

`.npy` 文件体积较大，浏览器直接加载会影响性能和稳定性。因此项目通过导出脚本抽取代表性样本，生成前端可读取的 JSON 文件。

### 问题 6：项目创新点是什么？

可以从三个方面回答：第一，将交通时序特征、事故标签和空间邻接关系结合到风险预测流程中；第二，使用 Attention-LSTM 建模交通风险的时间变化；第三，实现了从模型结果到前端可视化展示的完整闭环，并加入预测校验和解释分析。

### 问题 7：项目不足是什么？

当前系统仍是离线原型，尚未接入实时交通接口；前端展示采用代表性样本，不是全量数据实时展示；地图和区域语义还可以进一步接入真实 GIS 数据增强。

## 17. 答辩现场演示流程

答辩前先在项目根目录启动服务：

```powershell
python -m http.server 8000
```

打开浏览器：

```text
http://localhost:8000/dashboard_fixed.html
```

演示顺序建议：

1. 先展示 NYC 页面，说明风险热力图和时间轴。
2. 切换不同时间片，说明风险随时间变化。
3. 展示 TP、FP、FN 校验，说明模型预测与真实标签的关系。
4. 切换 Chicago 页面，说明系统支持双城展示。
5. 展示指标面板，说明模型效果。
6. 展示区域解释面板，说明空间邻接和风险概率。

现场如果页面无法打开，可以使用 PPT 中的截图兜底。推荐截图素材：

- `E:\VScodeProject\trafficcompare\assets\frontend-real-screenshot.png`
- `E:\VScodeProject\trafficcompare\assets\promo-dashboard-showcase.png`

## 18. 答辩中最稳妥的核心表述

可以记住下面这段话，答辩中很多问题都可以围绕它展开：

> 本系统使用 NYC 和 Chicago 交通时序数据、事故标签、空间邻接关系和模型评估结果完成训练与分析。前端为了保证交互性能，没有直接加载全量 `.npy` 文件，而是通过导出脚本抽取代表性时间帧，并生成前端可读 JSON 文件进行可视化展示。系统目前定位为离线预测与可视化原型，不是实时生产级交通平台。

## 19. PPT 制作注意事项

- 不要在 PPT 中堆大量代码。
- 尽量使用流程图、模型结构图、结果表格和前端截图。
- 实验结果页只放核心指标，不要把所有对比表全部放满。
- 一定说明系统边界：离线原型，不是实时系统。
- 指标解释时重点强调 AUC-PR、F1 和 Recall。
- 演示前确认 `E:\VScodeProject\trafficcompare\results\` 目录下 JSON 文件存在。
- 准备一页截图兜底，避免现场运行环境出问题。



