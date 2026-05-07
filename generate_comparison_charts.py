import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel文件
df = pd.read_excel('交通碰撞预测模型训练结果.xlsx')

# 清理数据：删除包含NaN的行，并将'-'替换为NaN
df_clean = df.dropna().replace('-', np.nan).dropna()

df_clean['配置'] = df_clean.apply(
    lambda row: f"{row['数据集']}\n{row['是否开启自适应平滑门控']}-{row['是否开启Streaming后处理']}",
    axis=1,
)

# 打印清理后的数据
print(df_clean)

# 定义指标列
metrics = ['F1分数', '召回率', '精确率', '准确率', 'AP', 'AUC']

# 1. 条形图：NYC和Chicago在不同配置下的指标对比
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('NYC和Chicago数据集在不同配置下的指标对比', fontsize=18)

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    sns.barplot(data=df_clean, x='配置', y=metric, hue='数据集', ax=ax, palette='Set2', dodge=False)
    ax.set_xlabel('配置', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} 对比', fontsize=13)
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.legend(loc='upper right', fontsize=10)

plt.subplots_adjust(bottom=0.24, hspace=0.4, wspace=0.35)
plt.savefig('dataset_config_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 1b. 横向条形图：NYC和Chicago在不同配置下的指标对比
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('横向条形：NYC和Chicago数据集在不同配置下的指标对比', fontsize=18)

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    sns.barplot(data=df_clean, y='配置', x=metric, hue='数据集', ax=ax, palette='Set2', dodge=False)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('配置', fontsize=11)
    ax.set_title(f'{metric} 横向对比', fontsize=13)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(loc='lower right', fontsize=10)

plt.subplots_adjust(left=0.26, hspace=0.4, wspace=0.35, bottom=0.1)
plt.savefig('dataset_config_comparison_horizontal.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 1c. 每个指标单独一个大图
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_clean, x='配置', y=metric, hue='数据集', ax=ax, palette='Set2', dodge=False)
    ax.set_xlabel('配置', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} 单指标对比', fontsize=16)
    ax.tick_params(axis='x', rotation=35, labelsize=10)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.legend(loc='upper right', fontsize=11)
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(f'dataset_config_{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# 2. 雷达图：不同配置的综合指标对比（以NYC为例）
def radar_plot(data, labels, title):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data = data.tolist()
    data += data[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.fill(angles, data, 'o-', alpha=0.25)
    ax.plot(angles, data, 'o-', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, size=16, fontweight='bold')
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# NYC数据
nyc_df = df_clean[df_clean['数据集'] == 'NYC']
for idx, row in nyc_df.iterrows():
    config = f"{row['是否开启自适应平滑门控']}-{row['是否开启Streaming后处理']}"
    values = row[metrics].values.astype(float)
    radar_plot(values, metrics, f'NYC {config} 雷达图')

# Chicago数据
chicago_df = df_clean[df_clean['数据集'] == 'Chicago']
for idx, row in chicago_df.iterrows():
    config = f"{row['是否开启自适应平滑门控']}-{row['是否开启Streaming后处理']}"
    values = row[metrics].values.astype(float)
    radar_plot(values, metrics, f'Chicago {config} 雷达图')

# 3. 热力图：所有配置的指标矩阵
try:
    pivot_df = df_clean.pivot_table(values=metrics, index=['数据集', '是否开启自适应平滑门控', '是否开启Streaming后处理'])
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('指标热力图', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        heatmap_data = pivot_df[metric].unstack(level=[1,2])
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', ax=ax)
        ax.set_title(f'{metric} 热力图')

    plt.tight_layout()
    plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
except Exception as e:
    print(f"热力图生成失败: {e}")

print("图表生成完成！")

# 4. 箱线图：指标分布对比
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('指标分布箱线图对比', fontsize=18)

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    sns.boxplot(data=df_clean, x='数据集', y=metric, hue='是否开启自适应平滑门控', ax=ax, palette='Set2')
    ax.set_xlabel('数据集', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} 分布', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_boxplot.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 5. 散点图：F1分数 vs 准确率，按配置分组
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_clean, x='F1分数', y='准确率', hue='配置', style='数据集', s=100, ax=ax, palette='tab10')
ax.set_xlabel('F1分数', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('F1分数 vs 准确率 散点图', fontsize=16)
ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('f1_vs_accuracy_scatter.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 6. 相关性热力图：指标之间的相关性
correlation_matrix = df_clean[metrics].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
ax.set_title('指标相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig('metrics_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 7. 差异条形图：配置变化对指标的影响
df_diff = df_clean.copy()
df_diff['配置组合'] = df_diff['是否开启自适应平滑门控'] + '-' + df_diff['是否开启Streaming后处理']
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('配置组合对指标的影响', fontsize=18)

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    sns.barplot(data=df_diff, x='配置组合', y=metric, hue='数据集', ax=ax, palette='Set2')
    ax.set_xlabel('配置组合', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} 配置影响', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('config_impact_bar.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("额外图表生成完成！")

# 8. 配对图：所有指标的散点图矩阵
fig = sns.pairplot(df_clean[metrics], diag_kind='kde', plot_kws={'alpha': 0.6})
fig.fig.suptitle('指标配对散点图矩阵', y=1.02, fontsize=16)
plt.savefig('metrics_pairplot.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 9. 直方图：每个指标的分布
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('指标分布直方图', fontsize=18)

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    sns.histplot(data=df_clean, x=metric, hue='数据集', ax=ax, kde=True, alpha=0.7, palette='Set2')
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('频次', fontsize=11)
    ax.set_title(f'{metric} 分布', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_histograms.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 10. 密度图：指标密度估计
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('指标密度估计图', fontsize=18)

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    for dataset in df_clean['数据集'].unique():
        subset = df_clean[df_clean['数据集'] == dataset]
        sns.kdeplot(data=subset, x=metric, ax=ax, label=dataset, fill=True, alpha=0.5)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.set_title(f'{metric} 密度', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_density.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("更多图表生成完成！")