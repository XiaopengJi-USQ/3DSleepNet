
import os
import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import Utils as utils
from model.Model import *
from DataLoader_ISRUC_S3_DownSampling import DataLoader
import time

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

preprocess_config = utils.ReadConfig('ISRUC_S3','Preprocess_ISRUC_S3')
train_config = utils.ReadConfig('ISRUC_S3','Train')
dataloader = DataLoader(preprocess_config,train_config)
model_save_path = train_config['save_path']
train_epoch = train_config['epoch']
train_lr = train_config['lr']
train_batch_size = train_config['batch_size']

all_scores = []
signal_file_list = [str(i).zfill(2) + "-1.npz" for i in range(1, 11)]
de_file_list = [str(i).zfill(2) + "-de.npz" for i in range(1, 11)]
psd_file_list = [str(i).zfill(2) + "-psd.npz" for i in range(1, 11)]
K.clear_session()

best_rs_path = './result/'

#de_psd_signal
def de_psd_signal_eval():
    for k in range(0,train_config['fold']):
        print(str(k)+"-th fold:")
        print("Read data")
        signal_np_data = np.load(os.path.join(preprocess_config['save_path'], signal_file_list[k]))
        de_np_data = np.load(os.path.join(preprocess_config['save_path'], de_file_list[k]))
        psd_np_data = np.load(os.path.join(preprocess_config['save_path'], psd_file_list[k]))

        val_signal = signal_np_data['data']
        val_de = de_np_data['data']
        val_psd = psd_np_data['data']
        val_label = signal_np_data['label']

        print("Read data successfully!")
        print(val_signal.shape)
        print(val_de.shape)
        print(val_psd.shape)
        print(val_label.shape)

        val_label = to_categorical(val_label, num_classes=5)

        model = Model_3DCNN_3stream(9,9,300,10)
        custom_objects = {'PDPAtt': PDPAtt}
        model = load_model(os.path.join(model_save_path, str(k)+'-best.h5'), custom_objects = custom_objects)
        val_loss, val_acc = model.evaluate(
            [val_signal,val_de,val_psd], val_label, verbose=0)

        predicts = model.predict([val_signal,val_de, val_psd])

        print('Evaluate', val_acc)
        all_scores.append(val_acc)
        AllPred_temp = np.argmax(predicts, axis=1)
        AllTrue_temp = np.argmax(val_label, axis=1)

        if k == 0:
            AllPred = AllPred_temp
            AllTrue = AllTrue_temp
        else:
            AllPred = np.concatenate((AllPred, AllPred_temp))
            AllTrue = np.concatenate((AllTrue, AllTrue_temp))
        # Fold finish
        print(128 * '_')
        del model,  val_signal, val_de, val_psd, val_label
    print(128 * '=')
    print("All folds' acc: ", all_scores)
    print("Average acc of each fold: ", np.mean(all_scores))

    # Print score to console
    print(128 * '=')

    utils.PrintScore(AllTrue, AllPred, all_scores, )



    # Print confusion matrix and save
    utils.ConfusionMatrix(AllTrue, AllPred, classes=['W', 'N1', 'N2', 'N3', 'REM'],
                          savePath='./result/')

    print('End of evaluating 3DCNN.')
    print(128 * '#')


de_psd_signal_eval()


