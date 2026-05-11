# Defense Quick Card

## One-Sentence Project Description

This project builds an offline traffic collision risk forecasting and visualization prototype based on Attention-LSTM, using NYC and Chicago traffic time-series data, crash labels, spatial adjacency relations, model evaluation results, and an interactive dashboard.

## Must-Say Boundary

This is an offline forecasting and visualization prototype, not a real-time production traffic platform.

## Key Numbers

| Dataset | Time Steps | Regions | Features | Samples | Positive Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| NYC | 13,128 | 64 | 116 | 840,192 | 22.53% |
| Chicago | 8,784 | 27 | 116 | 237,168 | 13.38% |

## Frontend Display Samples

| City | Frames | Regions | Records | Records with TP/FP/FN |
| --- | ---: | ---: | ---: | ---: |
| NYC | 13 | 64 | 832 | 296 |
| Chicago | 13 | 27 | 351 | 83 |

## Main Results

| Dataset | AUC-PR | AUC-ROC | F1 | Recall |
| --- | ---: | ---: | ---: | ---: |
| NYC | 0.6955 | 0.8786 | 0.6443 | 0.8265 |
| Chicago | 0.5617 | 0.8290 | 0.4730 | 0.7055 |

## Most Likely Questions

**Is the frontend data real?**

Core prediction probabilities, flow features, label verification, model metrics and topology relations come from local data and model exports. Some region descriptions and display text are visualization mappings.

**Why only 13 frames?**

The original `.npy` files are too large for direct browser loading. The frontend uses representative exported JSON samples for smooth interaction. Full training and evaluation are still done on backend data.

**Is it real-time?**

No. It is currently an offline prototype. Real-time traffic APIs and online updates are future work.

**Why focus on Recall?**

In traffic safety warning, missing a real risk is more costly than a false alarm, so recall is critical.

**Why can downloaded data look broken?**

Large `.npy` and `.h5` files are stored with Git LFS. Users must run `git lfs pull`; Download ZIP may only provide pointer files.

