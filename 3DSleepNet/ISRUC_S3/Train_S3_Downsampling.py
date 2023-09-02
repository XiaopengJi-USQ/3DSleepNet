import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import Utils as utils
from model.Model import *
from DataLoader_ISRUC_S3_DownSampling import DataLoader
import argparse

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
K.clear_session()
preprocess_config = utils.ReadConfig('Preprocess_ISRUC_S3')
train_config = utils.ReadConfig('Train')
dataloader = DataLoader(preprocess_config,train_config)
model_save_path = train_config['save_path']
train_epoch = train_config['epoch']
train_lr = train_config['lr']
train_batch_size = train_config['batch_size']

parser = argparse.ArgumentParser()
parser.add_argument("-k", type = int, default=10)
parser.add_argument("-i", type = int, default=0)
args = parser.parse_args()
i = args.i



def de_psd_signal_train():
    print(str(i)+"-th fold:")
    print("Read data")
    train_signal, train_de, train_psd, train_label, val_signal, val_de, val_psd, val_label = dataloader.Get_i_th_fold(i)
    print("Read data successfully!")
    train_label = to_categorical(train_label,num_classes=5)
    val_label = to_categorical(val_label,num_classes=5)
    model = Model_3DCNN_3stream(9,9,300,10)

    if i==0:
        print(model.summary())
    adam = tf.keras.optimizers.Adam(
        learning_rate=train_config['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])

    early_stopping = EarlyStopping(
            monitor='loss', patience=20, verbose=1)

    save_model = ModelCheckpoint(
            filepath=os.path.join(model_save_path, str(i)+'-best.h5'),
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
    history = model.fit([train_signal, train_de, train_psd],
                            train_label, epochs=train_epoch, batch_size=train_batch_size,
                            validation_data=([val_signal, val_de, val_psd], val_label),
                            callbacks=[early_stopping, save_model], verbose=2)
    del model, train_signal, train_de, train_psd, train_label, val_signal, val_de, val_psd, val_label

de_psd_signal_train()