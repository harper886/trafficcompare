import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

print('Starting...')
try:
    df = pd.read_excel('交通碰撞预测模型训练结果.xlsx')
    print('Excel loaded')
    df_clean = df.dropna().replace('-', float('nan')).dropna()
    print('Data cleaned')
    metrics = ['F1分数', '召回率', '精确率', '准确率', 'AP', 'AUC']
    print('Metrics defined')
    
    # 趋势图：用配置作为x轴
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('指标趋势图（按配置排序）', fontsize=18)
    
    # 排序配置
    config_order = ['关闭-关闭', '关闭-开启', '开启-关闭', '开启-开启']
    df_clean['config_order'] = df_clean.apply(lambda row: f"{row['是否开启自适应平滑门控']}-{row['是否开启Streaming后处理']}", axis=1)
    df_clean['config_num'] = df_clean['config_order'].map({c: i for i, c in enumerate(config_order)})
    
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        for dataset in df_clean['数据集'].unique():
            subset = df_clean[df_clean['数据集'] == dataset].sort_values('config_num')
            ax.plot(subset['config_num'], subset[metric], marker='o', label=dataset)
        ax.set_xlabel('配置', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} 趋势', fontsize=13)
        ax.set_xticks(range(len(config_order)))
        ax.set_xticklabels(config_order, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('metrics_trend.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('趋势图 done')
    
    # 堆叠图：堆叠条形图
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('指标堆叠图（配置贡献）', fontsize=18)
    
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        df_pivot = df_clean.pivot(index='数据集', columns='config_order', values=metric)
        df_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_xlabel('数据集', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} 堆叠', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('metrics_stacked.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('堆叠图 done')
    
    # 3D可视化：3D散点图
    print('Creating 3D scatter...')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'NYC': 'red', 'Chicago': 'blue'}
    for dataset in df_clean['数据集'].unique():
        subset = df_clean[df_clean['数据集'] == dataset]
        ax.scatter(subset['F1分数'], subset['准确率'], subset['AP'], c=colors[dataset], label=dataset, s=100)
    ax.set_xlabel('F1分数')
    ax.set_ylabel('准确率')
    ax.set_zlabel('AP')
    ax.set_title('3D散点图：F1分数 vs 准确率 vs AP')
    ax.legend()
    plt.savefig('metrics_3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('3D图 done')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()