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

import tensorflow as tf
import numpy as np
from get_classes import get_classes
from src.utils import normalize_data, shuffle_data
from tools import calculate_confusion_matrix, plot_confusion_matrix, calculate_acc_cm_each_snr, plot_results_comparison

# from src.encoder_model import TransformerBlock, SoftThresholding

random.seed(2016)  
np.random.seed(2016) 
tf.random.set_seed(2016)

# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./figure'):
    os.makedirs('./figure')

if not os.path.exists('./figure/benign'):
    os.makedirs('./figure/benign')

if not os.path.exists('./figure/poison'):
    os.makedirs('./figure/poison')

def model_predict(batch_size=400,
                  weights_path=None,
                  tune_weight_path = "./saved_models/weight_tune.hdf5",
                  min_snr=-20,
                  test_datapath='./data/rml16_test_1_10_data.hdf5',
                  classes_path='./data/classes_rml16.txt',
                  tune_save_plot_file='./figure/GRU2_total_confusion.png'):
    
    classes = get_classes(classes_path)
    n_classes = len(classes)

    test_data = h5py.File(test_datapath, 'r')
    X_test = test_data['X_test'][:, :, :]
    Y_test = test_data['Y_test'][:].astype(np.int32)

    # 归一化测试数据
    # X_test = tensor_normalize_data(X_test)

    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, num_classes=n_classes)
    ###############################################################################################
    # GRU / ResNet
    tune_model = tf.keras.models.load_model(tune_weight_path)
    # RSNet
    # tune_model = tf.keras.models.load_model(tune_weight_path,
    #                                        custom_objects={'SoftThresholding': SoftThresholding})
    # Transformer
    # tune_model = tf.keras.models.load_model(tune_weight_path,
    #                                        custom_objects={'TransformerBlock': TransformerBlock})
    ###############################################################################################

    tune_test_Y_predict = tune_model.predict(X_test, batch_size=batch_size, verbose=1)
    # 计算混淆矩阵
    confusion_matrix_normal, right, wrong = calculate_confusion_matrix(Y_test_categorical, tune_test_Y_predict, classes)
    overall_accuracy = round(1.0 * right / (right + wrong), 4)
    print('Overall Accuracy: %.2f%% / (%d + %d)' % (100 * overall_accuracy, right, wrong))
    with open('./figure/benign/benign_tune_results.txt', 'a') as file:
        file.write('Overall Accuracy: %.2f%% / (%d + %d)\n' % (100 * overall_accuracy, right, wrong))
    plot_confusion_matrix(confusion_matrix_normal, labels=classes, save_filename=tune_save_plot_file)
    # calculate_acc_cm_each_snr(Y_test_categorical, tune_test_Y_predict, Z_test, classes, min_snr=min_snr, file_name="tune")


    test_data.close()


if __name__ == '__main__':
    model_predict(
        batch_size=400,
        weights_path=None,

        tune_weight_path = "./saved_models/poisoned_weight_tune.hdf5",
        # tune_weight_path = "./saved_models/weight_tune.hdf5",
        # tune_weight_path = "./saved_models/weight_sup.hdf5",
        # tune_weight_path = "./target/clean/weight_tune.hdf5",

        min_snr=-20,
        test_datapath='./data/rml25_test_1_10_data_9.hdf5',
        classes_path='./data/classes_rml25_9.txt',
        tune_save_plot_file='./figure/benign/benign_tune_total_confusion.png',
    )
    