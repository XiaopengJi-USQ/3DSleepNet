#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import Utils
import os

class DataLoader():

    def __init__(self, preprocess_config, train_config):
        self.preprocess_config = preprocess_config
        self.train_config = train_config

        self.signal_file_list = [str(i).zfill(2)+"-1.npz" for i in range(1,11)]
        self.de_file_list = [str(i).zfill(2)+"-de.npz" for i in range(1,11)]
        self.psd_file_list = [str(i).zfill(2)+"-psd.npz" for i in range(1,11)]
        print(self.signal_file_list)
        print(self.de_file_list)
        print(self.psd_file_list)


    def Get_i_th_fold(self,i):
        val_first = True
        train_first = True
        for j in range(0, len(self.signal_file_list)):
            signal_np_data = np.load(os.path.join(self.preprocess_config['save_path'],self.signal_file_list[j]))
            de_np_data = np.load(os.path.join(self.preprocess_config['save_path'],self.de_file_list[j]))
            psd_np_data = np.load(os.path.join(self.preprocess_config['save_path'],self.psd_file_list[j]))
            signal_data = signal_np_data['data']
            de_data = de_np_data['data']
            psd_data = psd_np_data['data']
            label = signal_np_data['label']
            #val_set
            print("index=",j,"subject=",self.signal_file_list[j])
            if ((self.signal_file_list[j][0:-6] == str(i + 1).zfill(2) )):
                if val_first:
                    print("1st Val:",self.signal_file_list[j])
                    val_de = de_data
                    val_psd = psd_data
                    val_signal = signal_data
                    val_label = label
                    val_first = False
            else:
                # train_set
                if train_first:
                    train_de = de_data
                    train_psd = psd_data
                    train_signal = signal_data
                    train_label = label
                    train_first = False
                else:
                    train_de = np.concatenate([train_de, de_data], axis=0)
                    train_psd = np.concatenate([train_psd, psd_data], axis=0)
                    train_signal = np.concatenate([train_signal, signal_data], axis=0)
                    train_label = np.concatenate([train_label, label], axis=0)
        return train_signal, train_de, train_psd, train_label, val_signal, val_de, val_psd, val_label

if __name__ == '__main__':
    preprocess_config = Utils.ReadConfig('ISRUC_S3','Preprocess_ISRUC_S3')
    train_config = Utils.ReadConfig('ISRUC_S3','Train')
    dataloader = DataLoader(preprocess_config,train_config)
    train_signal, train_de, train_psd, train_label, val_signal, val_de, val_psd, val_label = dataloader.Get_i_th_fold(2)
    print("train_signal.shape",train_signal.shape)
    print("train_de.shape",train_de.shape)
    print("train_psd.shape",train_psd.shape)
    print("train_label.shape",train_label.shape)
    print("val_signal.shape",val_signal.shape)
    print("val_de.shape",val_de.shape)
    print("val_psd.shape",val_psd.shape)
    print("val_label.shape",val_label.shape)
