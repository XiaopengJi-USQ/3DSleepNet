# 3DSleepNet

Code of the paper '3DSleepNet: A Multi-Channel Bio-Signal Based Sleep Stages Classification Method Using Deep Learning'.
## Dataset

The ISRUC dataset can be downloaded on the official website: https://sleeptight.isr.uc.pt/?page_id=76

## How to run
### 1. Modify configuration files

Modify configuration files especially paths in each config file.
    
    The `original_data_path` item in `Dataset_ISRUC_S3.json` is the path of .mat files
    The `label_path` item in `Dataset_ISRUC_S3.json` is the path of .txt label files
    The `save_path` item in `Prepocess_ISRUC_S3.json` is the path to save preprocessed data.
    The `save_path` item in `Train.json` is the path to save the trained model files.

All paths above are folders.

### 2. Extract Features

Run `Feature_extract_ISRUC_S3.py` by  `python train_FeatureNet.py`

This program will extract features and save features to the folder 'extracted_features'.

Run `Downsampling.py` by `python Downsampling.py`

This program will downsample time series and save downsampled data to the folder 'extracted_features'.

### 3. Train

Run `run.bat`

This program will train models and save them to the folder 'result'.

### 4. Evaluate

Run `Evaluate_S3_downsampling.py` by `python Evaluate_S3_downsampling.py`

This program will evaluate models and save them to the folder 'result'.
