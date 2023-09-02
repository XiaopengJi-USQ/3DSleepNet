import numpy as np
import Utils
import os


def S3_DownSampling():
    preprocessed_folder = "./extracted_features/"
    saveparth = "./extracted_features/"
    for i in range(1,11):
        print("downsampling subject:",i)
        file_path = os.path.join(preprocessed_folder, str(i).zfill(2)+"-1.npz")
        npfile = np.load(file_path)
        data = npfile['data']
        label = npfile['label']
        print("original data shape:", data.shape)
        split = 10
        data.resize((data.shape[0], data.shape[1], data.shape[2],int(data.shape[3]/split),split))
        print("resized data shape:", data.shape)
        # print(data)
        new_data = np.mean(data, axis=4)
        new_data.resize((data.shape[0], data.shape[1], data.shape[2], data.shape[3] ))
        print("new data shape:", data.shape)
        print("label shape:", label.shape)
        # print(new_data)
        new_file_path = os.path.join(saveparth,str(i).zfill(2)+"-1.npz")
        np.savez(new_file_path,data = new_data, label = label)

S3_DownSampling()