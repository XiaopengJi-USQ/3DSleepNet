#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import Utils
import os
import scipy.io as sio
from Preprocess import *
from Feature import *
import time


class Feature_extract_ISRUC_S3(Preprocess):


    def __init__(self):
        super().__init__()
        self.dataset_config = Utils.ReadConfig('ISRUC_S3','Dataset_ISRUC_S3')
        self.preprocess_config = Utils.ReadConfig('ISRUC_S3','Preprocess_ISRUC_S3')
        self.data_file_list = Utils.GetFileList(self.dataset_config['original_data_path'], '.mat',
                                                self.preprocess_config['exclude_subjects_data'])
        self.label_file_list = Utils.GetFileList(self.dataset_config['label_path'], '.txt',
                                                 self.preprocess_config['exclude_subjects_label'])
        self.data_file_list.sort(key=lambda x:int(x[0:2]))
        self.label_file_list.sort(key=lambda x:int(x[0:2]))


    def Read1DataFile(self,file_name):
        file_path = os.path.join(self.dataset_config['original_data_path'],file_name)
        mat_data = sio.loadmat(file_path)
        start = True
        for each_channel in self.preprocess_config['channels_to_use']:
            if start:
                original_data = [mat_data[each_channel]]
                start = False
            else:
                original_data = np.concatenate((original_data,[mat_data[each_channel]]))
        return original_data.reshape((original_data.shape[0],original_data.shape[1]*original_data.shape[2]))

    def Read1LabelFile(self,file_name):
        file_path = os.path.join(self.dataset_config['label_path'], file_name)
        original_label = list()
        with open(file_path, "r") as f:
            for line in f.readlines():
                if (line != '' and line != '\n'):
                    label = int(line.strip('\n'))
                    original_label.append(label)
        return original_label


    def FormatLabel2Epochs(self,epoch_length,labels,overlap_percent=0):
        copy_numbers = int(30/epoch_length)
        format_labels = list()

        if overlap_percent==0:

            for i in labels:
                if (i == 5) or (i == '5'):
                    i = 4
                for j in range(0,copy_numbers):
                    format_labels.append(i)
        else:
            None
        return np.array(format_labels[:-30*copy_numbers])

    def Data_Split(self, split_number, sampling_rate, epoch_length, signle_channel_data):
        data_point_number = signle_channel_data.shape[0]
        epoch_number = int(data_point_number/epoch_length/sampling_rate)
        data_point_after_split = int(data_point_number/split_number)
        split_data = np.zeros([split_number,data_point_after_split], dtype = float)
        for i in range(0, data_point_after_split):
            for j in range(0, split_number):
                split_data[j,i] = signle_channel_data[split_number * i+j]

        return split_data



if __name__ == '__main__':
    feature_extract = Feature_extract_ISRUC_S3()
    split_number = 2
    extract_wave_config = feature_extract.preprocess_config['extract_frequency_bands']
    feature = Feature()
    stft_f = {
            'stftn': 6000,
            'fStart': [0.5, 2, 4, 6, 8, 11, 14, 22, 31],
            'fEnd': [4, 6, 8, 11, 14, 22, 31, 40, 50],
            'fs': 200,
            'window': 30,
        }
    stft_tf = {
            'stftn': 600,
            'fStart': [0.5, 2, 4, 6, 8, 11, 14, 22, 31],
            'fEnd': [4, 6, 8, 11, 14, 22, 31, 40, 50],
            'fs': 200,
            'window': 3,
        }

    for i in range(0,len(feature_extract.data_file_list)):
        print("Process subject:",i)
        data = feature_extract.Read1DataFile(feature_extract.data_file_list[i])
        channel_numbers = data.shape[0]
        de_length = 600
        psd_length = 6000
        de_frames = 10
        psd_frames = 1
        epoch_number = int(data.shape[1] / feature_extract.preprocess_config['sampling_rate'] / feature_extract.preprocess_config['epoch_length'])
        reshaped_data_de = data.reshape((channel_numbers,epoch_number* de_frames,de_length))
        reshaped_data_de = np.transpose(reshaped_data_de,[1,0,2])

        reshaped_data_psd = data.reshape((channel_numbers,epoch_number* psd_frames,psd_length))
        reshaped_data_psd = np.transpose(reshaped_data_psd,[1,0,2])
        print("reshaped_data.shape",", de.shape:",reshaped_data_de.shape,", psd.shape:",reshaped_data_psd.shape)
        de_rs = list()
        psd_rs = list()
        starttime = time.perf_counter()
        #time frequency features
        for r in range(0, epoch_number*de_frames):
            _, de = feature.CalDe_Psd(reshaped_data_de[r],stft_tf)
            de_rs.append(de)
        #frequency features
        for r in range(0, epoch_number*psd_frames):
            psd, _ = feature.CalDe_Psd(reshaped_data_psd[r],stft_f)
            psd_rs.append(psd)

        de_rs = np.array(de_rs)
        psd_rs = np.array(psd_rs)

        de_rs = np.reshape(de_rs, (epoch_number,channel_numbers,-1,de_frames))
        psd_rs = np.reshape(psd_rs, (epoch_number,channel_numbers,-1,psd_frames))
        np.savez(os.path.join(feature_extract.preprocess_config['save_path'], str(i+1).zfill(2)+"-de"), data = de_rs)
        np.savez(os.path.join(feature_extract.preprocess_config['save_path'], str(i+1).zfill(2)+"-psd"), data = psd_rs)

        channels = list()
        for each_channel in range(0, channel_numbers):
            data_split_each_channel = feature_extract.Data_Split(split_number,feature_extract.preprocess_config['sampling_rate'],feature_extract.preprocess_config['epoch_length'], data[each_channel])
            waves = list()
            for split in range(0, data_split_each_channel.shape[0]):
                extract_waves = feature_extract.ExtractWaves(100,extract_wave_config,data_split_each_channel[split])
                waves.append(extract_waves)
            channels.append(waves)
        channels = np.array(channels)
        channels = np.reshape(channels,(channels.shape[0],channels.shape[1],channels.shape[2],epoch_number,int(feature_extract.preprocess_config['sampling_rate']*feature_extract.preprocess_config['epoch_length']/split_number)))
        formatted_data = np.transpose(channels, [1,3,0,2,4])


        for j in range(0, formatted_data.shape[0]):
            if j==0:
                print("Save subject ", str(i + 1).zfill(2) + "-" + str(j + 1))
                label = feature_extract.Read1LabelFile(feature_extract.label_file_list[i])
                label[label == 5] = 4
                # label = label[:-30]
                formatted_label = feature_extract.FormatLabel2Epochs(feature_extract.preprocess_config['epoch_length'], label)
                feature_extract.SavePreprocessedData(feature_extract.preprocess_config['save_path'],
                                                str(i + 1).zfill(2) + "-" + str(j + 1), formatted_data[j], formatted_label)

    print("Preprocess complete!")

