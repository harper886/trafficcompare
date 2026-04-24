
from model import MYPLAN
# ... (这里需要你模型实例化的代码)
model.load_weights('checkpoint_nyc_epoch_0.h5')
print("加载成功！文件没损坏。")