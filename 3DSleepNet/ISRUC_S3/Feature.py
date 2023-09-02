# -*- coding: utf-8 -*-
import numpy as np
from scipy.fftpack import fft, ifft
import math
import itertools as it
import scipy.stats
from mne import filter


class Feature:


    def CalDe_Psd(self, signals, stft_para):

        STFTN = stft_para['stftn']
        fStart = stft_para['fStart']
        fEnd = stft_para['fEnd']
        fs = stft_para['fs']
        window = stft_para['window']



        fStartNum = np.zeros([len(fStart)], dtype=int)
        fEndNum = np.zeros([len(fEnd)], dtype=int)
        for i in range(0, len(stft_para['fStart'])):
            fStartNum[i] = int(fStart[i] / fs * STFTN)
            fEndNum[i] = int(fEnd[i] / fs * STFTN)

        n = signals.shape[0]
        m = signals.shape[1]

        psd = np.zeros([n, len(fStart)])
        de = np.zeros([n, len(fStart)])


        Hlength = window * fs


        Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])


        dataNow = signals[0:n]

        for j in range(0, n):

            temp = dataNow[j]

            Hdata = temp * Hwindow
            FFTdata = fft(Hdata, STFTN)
            magFFTdata = abs(
                FFTdata[0:int(STFTN / 2)])

            for p in range(0, len(fStart)):

                E = 0
                for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                    E = E + magFFTdata[p0] * magFFTdata[p0]

                E = E / (fEndNum[p] - fStartNum[p] + 1)
                psd[j][p] = E
                de[j][p] = math.log(100 * E, 2)
        return psd, de

    def SaveFeatures(self, data, savepath, featurename='DE_PSD'):
        saved_data = np.save(savepath + featurename, data)
        print("save path:" + savepath + featurename)