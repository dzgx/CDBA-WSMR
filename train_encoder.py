import os,random,h5py,warnings
# 过滤掉INFO、WARNING和ERROR级别的日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 启用确定性操作模式
os.environ['TF_DETERMINISTIC_OPS'] = '1' 
# 禁用OneDNN优化以确保完全的确定性
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 指定使用的GPU设备ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from src.dataset2016 import load_data
from statistics import mean
from src.utils import *
from get_classes import get_classes
random.seed(2016)  
np.random.seed(2016) 
tf.random.set_seed(2016)

######################################################################################################
 # 加载数据集
n_classes = len(get_classes(from_file="./data/classes_rml25_9.txt"))
train_data = h5py.File('./data/rml25_train_8_10_data_9.hdf5', 'r')

X_train = train_data['X_train'][:, :, :]
Y_train = train_data['Y_train'][:].astype(np.int32)

######################################################################################################
shuffle_data(X_train, Y_train)
######################################################################################################
print("# of all train data:", X_train.shape)
print("# of all train label:", Y_train.shape)
######################################################################################################
# SimCLR
# sim_model, epoch_losses = train_simclr(X_train, batch_size=512, Epoch=100, temperature = 0.1)
sim_model, epoch_losses = train_simclr(X_train, Y_train, batch_size=900, Epoch=100, temperature = 0.1)
plot_epoch_loss()
######################################################################################################
train_data.close()
######################################################################################################