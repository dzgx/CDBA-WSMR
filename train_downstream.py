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
val_data = h5py.File('./data/rml25_val_1_10_data_9.hdf5', 'r')

X_train = train_data['X_train'][:, :, :]
X_val = val_data['X_val'][:, :, :]

Y_train = train_data['Y_train'][:].astype(np.int32)
Y_val = val_data['Y_val'][:].astype(np.int32)

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=n_classes)
Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=n_classes)

######################################################################################################
# 设置有标签数据数量
ratio_train_labeled = 0.1 # ratio of labeled training samples
ratio_val_labeled = 1.0   # ratio of labeled validation samples
# 对有标签数据进行分层抽样
X_train_labeled, Y_train_labeled = stratified_sampling(X_train, Y_train, ratio_train_labeled, n_modulations=n_classes, n_snr_levels=1)
X_val_labeled, Y_val_labeled = stratified_sampling(X_val, Y_val, ratio_val_labeled, n_modulations=n_classes, n_snr_levels=1)
######################################################################################################
# 打乱并归一化
shuffle_data(X_train_labeled, Y_train_labeled)

shuffle_data(X_val_labeled, Y_val_labeled)
######################################################################################################
print("# of all X_train labeled data:", X_train_labeled.shape)
print("# of all X_val labeled data", X_val_labeled.shape)
print("# of all Y_train labeled data:", Y_train_labeled.shape)
print("# of all Y_val labeled data", Y_val_labeled.shape)
######################################################################################################
# Tune Model
tune_model = train_tune(X_train_labeled, Y_train_labeled, X_val_labeled, Y_val_labeled)
######################################################################################################
train_data.close()
val_data.close()
######################################################################################################
