import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel('交通碰撞预测模型训练结果.xlsx')
df_clean = df.dropna().replace('-', float('nan')).dropna()

# 3D可视化：3D散点图
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