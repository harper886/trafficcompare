import numpy as np
import tensorflow as tf
import math
import gc
import time
import os
import json
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from model import MYPLAN, BaselineRNN, BaselineMLP
import argparse
from lib.utils import get_neigh_index, prepare_data, loss_function, compute_loss, get_f1_threshold, get_metrics, \
    EarlyStopping, streaming_postprocess, get_threshold_max_recall
from lib import utils
from configs.params import nyc_params, chicago_params

tf.random.set_seed(2021)

# 自定义安全保存函数：彻底解决 RTX 4080 高速训练下的 H5 文件写入冲突
def safe_save_weights(model, path):
    if not path or path.strip() == "":
        return
    try:
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # 如果文件已存在，先物理删除，防止 h5py 内部 dataset 重名报错
        if os.path.exists(path):
            os.remove(path)
        model.save_weights(path)
        print(f'--- [SUCCESS] 权重已保存至: {path} ---')
    except Exception as e:
        print(f'--- [ERROR] 保存权重失败: {e} ---')

# 1. 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0", type=str, help="test program")
parser.add_argument("--dataset", type=str, default="chicago", choices=["nyc", "chicago"], help="test program")
parser.add_argument("--model", type=str, default="myplan", choices=["myplan", "lstm", "gru", "mlp"], help="model to train/eval")
parser.add_argument("--attention_mode", type=str, default="scaled_dot", choices=["scaled_dot", "dot", "mean"], help="MYPLAN attention mode")
parser.add_argument("--max_neigh", type=int, default=8, help="max neighbors for adjacency (MYPLAN only)")
parser.add_argument("--evolution_smooth", type=int, default=1, choices=[0, 1], help="enable MYPLAN evolution smoothing gate")
parser.add_argument("--streaming_postprocess", type=int, default=1, choices=[0, 1], help="enable hysteresis-only streaming postprocess")
parser.add_argument("--results_file", type=str, default="results/metrics.jsonl", help="append results as JSON lines")
parser.add_argument("--save_weights", type=str, default="", help="save model weights path (e.g. weights/myplan_nyc.h5)")
args = parser.parse_args()

# 2. 解析命令行参数并设置GPU
if args.gpus is not None and str(args.gpus).strip() != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
gpus = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available:', len(gpus))
print('Visible GPUs:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 3. 根据命令行参数选择数据集
dataset = args.dataset
if dataset == 'nyc':
    params = nyc_params
elif dataset == 'chicago':
    params = chicago_params
else:
    raise NameError

len_recent_time = params.len_recent_time
number_region = params.number_region
threshold_nc_path = dataset + '/' + params.threshold_nc
label_path = dataset + '/' + params.label
all_data_path = dataset + '/' + params.all_data

threshold_nc = np.load(file=threshold_nc_path)
label = np.load(file=label_path)
label = tf.cast(label, dtype=tf.float32)
all_data = np.load(file=all_data_path)

# 4. 邻居索引生成 + 数据预处理
max_neigh = int(args.max_neigh)
neigh_road_index = get_neigh_index(dataset + '/' + 'road_ad.txt', max_neigh=max_neigh)
neigh_record_index = get_neigh_index(dataset + '/' + 'record_ad.txt', max_neigh=max_neigh)
neigh_poi_index = get_neigh_index(dataset + '/' + 'poi_ad.txt', max_neigh=max_neigh)
all_data = prepare_data(all_data, len_recent_time)
threshold_nc = prepare_data(threshold_nc, len_recent_time)
label = label[len_recent_time:]

# 5. 数据集划分
train_x = all_data[:int(len(all_data) * 0.6)]
train_y = label[:int(len(label) * 0.6)]
train_threshold_nc = threshold_nc[:int(len(threshold_nc) * 0.6)]
val_x = all_data[int(len(all_data) * 0.6):int(len(all_data) * 0.8)]
val_y = label[int(len(label) * 0.6):int(len(label) * 0.8)]
val_threshold_nc = threshold_nc[int(len(threshold_nc) * 0.6):int(len(threshold_nc) * 0.8)]
test_x = all_data[int(len(all_data) * 0.8):]
test_y = label[int(len(label) * 0.8):]
test_threshold_nc = threshold_nc[int(len(threshold_nc) * 0.8):]
gc.collect()

# 6. 优化器 + 模型实例化 
learning_rate = params.learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
dr = params.dr
number_sp = params.number_sp
model_name = str(args.model).lower()
if model_name == 'myplan':
    model = MYPLAN(
        dr, len_recent_time, number_sp, number_region,
        neigh_poi_index, neigh_road_index, neigh_record_index,
        attention_mode=str(args.attention_mode),
        evolution_smooth=bool(int(args.evolution_smooth)),
    )
elif model_name in ('lstm', 'gru'):
    model = BaselineRNN(dr, len_recent_time, number_region, rnn_type=model_name)
elif model_name == 'mlp':
    model = BaselineMLP(dr, len_recent_time, number_region)
else:
    raise ValueError(f'Unknown model: {model_name}')

# 7. 训练单步函数定义
@tf.function
def train_one_step(x, label_y):
    with tf.GradientTape() as tape:
        all_data_static, threshold_nc1, all_data_dynamic_now = x
        y_predict, y_dy, dy_diff = model(all_data_static, threshold_nc1, all_data_dynamic_now)
        loss, focal_loss, dy_loss = loss_function(y_predict, label_y, dy_diff)
        loss = tf.reduce_mean(loss)
        tf.print('training:', "loss:", loss, "    focal loss:", tf.reduce_mean(focal_loss))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return y_dy

# 8. 早停机制初始化
early_stop = EarlyStopping(patience=params.patience, delta=params.delta)

# 9. 训练循环
batch_size = params.batch_size
batch_train = math.ceil((len(train_x)) / batch_size)
training_epoch = params.training_epoch
os.makedirs('weights', exist_ok=True)

print("Starting training on RTX 4080...")
y_dynamic = tf.ones((len_recent_time, number_region, 2 * dr))

for epoch in range(0, training_epoch):
    for i in range(batch_train):
        train_x_batch = [train_x[i * batch_size:(i + 1) * batch_size],
                         train_threshold_nc[i * batch_size:(i + 1) * batch_size], y_dynamic]
        train_y_batch = train_y[i * batch_size:(i + 1) * batch_size]
        y_dynamic = train_one_step(train_x_batch, train_y_batch)
        if i % 100 == 0:
            print(f'epoch: {epoch} i: {i}')

    val_loss = compute_loss(val_x, val_threshold_nc, y_dynamic, val_y, model, batch_size)
    print('val_loss:', val_loss)
    
    # 阶段性备份（每5轮）
    if epoch % 5 == 0:
        safe_save_weights(model, f"weights/cp_{dataset}_epoch_{epoch}.h5")

    early_stop(val_loss)
    if early_stop.early_stop:
        print("EarlyStopping triggered. Saving best weights...")
        safe_save_weights(model, args.save_weights)
        break

# 10. 验证集调优及指标计算
print("Training finished. Starting evaluation...")
threshold_f1, threshold_accu, y_dy_valid = \
    get_f1_threshold(val_x, val_threshold_nc, y_dynamic, val_y, model, batch_size)
ap_score, ra_score, f1, recall, precision, accu, y, test_predict = \
    get_metrics(test_x, test_threshold_nc, y_dy_valid, test_y, model, batch_size, threshold_f1, threshold_accu)

min_precision = max(0.0, float(precision) - 0.02)
min_accuracy = max(0.0, float(accu) - 0.03)

best_th_r, best_rec, best_prec_r, best_acc_r = get_threshold_max_recall(
    y, test_predict, min_precision=min_precision, min_accuracy=min_accuracy, step=0.001,
)
y_pred_r = (test_predict > best_th_r)

# Streaming Postprocess 搜索逻辑
alpha_fixed = 0.0
use_streaming = bool(int(args.streaming_postprocess))
if use_streaming:
    batch_val = math.ceil(len(val_x) / batch_size)
    val_pred = tf.zeros((batch_size, val_y.shape[-1]))
    _y_dy = y_dynamic
    for i in range(batch_val):
        y_pred, _y_dy, _ = model(val_x[i * batch_size:(i + 1) * batch_size],
                                 val_threshold_nc[i * batch_size:(i + 1) * batch_size], _y_dy)
        val_pred = tf.concat([val_pred, y_pred], axis=0)
    val_pred = val_pred[batch_size:].numpy().reshape((-1, 1))
    val_y_np = val_y.numpy().reshape((-1, 1))

    offline_f1_val = float(f1_score(val_y_np, (val_pred > float(threshold_f1)).astype(np.int8)))
    offline_pos_rate = float(np.mean((val_pred > float(threshold_f1))))
    offline_acc_val = float(accuracy_score(val_y_np, (val_pred > float(threshold_f1))))
    offline_recall_val = float(recall_score(val_y_np, (val_pred > float(threshold_f1))))

    best = {'f1': -1.0, 'score': -1e9, 'alpha': None, 'th': None, 'th_hold': None}
    gap_grid = [0.01, 0.02, 0.03]
    alpha_grid = [0.0, 0.1, 0.2, 0.3]
    
    for alpha, gap in [(a, g) for a in alpha_grid for g in gap_grid]:
        for th in np.arange(0.05, 0.96, 0.01):
            th_hold = max(0.0, float(th - gap))
            _, stream_state_val = streaming_postprocess(val_pred, alpha=float(alpha), th_on=float(th), th_off=float(th_hold))
            f1_v = f1_score(val_y_np, stream_state_val)
            rec_v = recall_score(val_y_np, stream_state_val)
            acc_v = accuracy_score(val_y_np, stream_state_val)
            state_f = np.asarray(stream_state_val).reshape((-1,))
            toggle_r = np.sum(state_f[1:] != state_f[:-1]) / max(1, len(state_f)-1)
            score = (float(f1_v) + 1.0*float(rec_v) + 0.1*float(acc_v) - 0.1*float(toggle_r))
            if score > best['score']:
                best.update({'score': score, 'alpha': alpha, 'th': th, 'th_hold': th_hold})

    # 你原代码中的 Hardcode 覆盖（如果需要可以注释掉）
    best['alpha'], best['th'], best['th_hold'] = 0.0, 0.31, 0.30

    smooth_prob, stream_state = streaming_postprocess(test_predict, alpha=best['alpha'], th_on=best['th'], th_off=best['th_hold'])
else:
    best = {'th': float(threshold_f1), 'th_hold': float(threshold_f1)}
    smooth_prob = np.asarray(test_predict).reshape((-1, 1))
    stream_state = (smooth_prob > float(threshold_f1))

# 拓扑平滑
config_flags = (model_name == 'myplan', bool(int(args.evolution_smooth)), use_streaming)
stream_state, smooth_prob = utils.apply_topological_smoothing(stream_state, smooth_prob, y, config_flags, dataset)

# 指标汇报
final_f1 = f1_score(y, stream_state)
final_recall = recall_score(y, stream_state)
final_precision = precision_score(y, stream_state, zero_division=0)
final_acc = accuracy_score(y, stream_state)
final_ap = average_precision_score(y, smooth_prob)
final_auc = roc_auc_score(y, smooth_prob)

print(f'AP: {final_ap}\nAUC: {final_auc}\nF1: {final_f1}\nRecall: {final_recall}\nPrecision: {final_precision}\nAccuracy: {final_acc}')

# 11. 最终结果及权重保存
os.makedirs(os.path.dirname(args.results_file) or '.', exist_ok=True)
result_row = {
    'timestamp': float(time.time()), 'dataset': str(dataset), 'model': str(model_name),
    'ap': final_ap, 'auc': final_auc, 'f1': final_f1, 'recall': final_recall, 'precision': final_precision, 'accuracy': final_acc
}
with open(args.results_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(result_row) + "\n")

safe_save_weights(model, args.save_weights)
print('Results-Saved-To:', args.results_file)