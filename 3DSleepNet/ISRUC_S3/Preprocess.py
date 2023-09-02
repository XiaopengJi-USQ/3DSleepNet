#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import Utils
import os
import scipy.io as sio
from scipy import signal
import pywt
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from EDFlib.edfwriter import EDFwriter
import sys

class Preprocess():


    def __init__(self):
        None

    def ExtractWaves(self,sampling_rate ,bandpass_config, to_extract_signle_channel_data):
        waves = list()

        for bandpass in bandpass_config:
            if bandpass_config[bandpass]['use']:
                Highcut = float(bandpass_config[bandpass]['high_cut'])
                Lowcut = float(bandpass_config[bandpass]['low_cut'])
                b, a = signal.butter(float(bandpass_config[bandpass]['level']),
                                     [2 * Lowcut / sampling_rate, 2 * Highcut / sampling_rate],
                                     'bandpass')
                wave = signal.filtfilt(b, a, to_extract_signle_channel_data)
                waves.append(wave)
        return np.array(waves)

    def SavePreprocessedData(self, path_to_save, file_name, data_to_save, label_to_save):
        try:
            np.savez(os.path.join(path_to_save,file_name), data = data_to_save, label = label_to_save)
            print("Save data", file_name, "successfully!")
        except BaseException:
            print("Save data", file_name, "fail")
