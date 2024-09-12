# FastECG
# A Pytorch implementation of our paper Computation-Efficient Semi-Supervised Learning for ECG-based Cardiovascular Diseases Detection
# [Arxiv](https://arxiv.org/pdf/2406.14377)
# Preliminaries
* Four downsteam datasets: The Chapman-shaoxing dataset, the PTB-XL dataset, the Ningbo dataset, the G12EC dataset.
* If you want to retrain the backbone, please download the CODE-15% via https://zenodo.org/records/4916206.
* quick download the four downsteam datasets: wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
* The pretrained base backbone (base_checkpoint.pkl). The medium and large backbones are too large to upload, we will provide the link to access them later on.
* Requirements: you can use (pip install -r requirements.txt) to install all the related packages used in our paper.
# FastECG Pre-Training on the CODE-15% or CODE-full Datasets.
* First, you should apply and download the CODE-15% dataset or the full CODE dataset
* For the full CODE dataset, you should collect all the ECG signals and labels in a h5py files (CODEfull_data.hdf5, CODEfull_labels.hdf5). There should be a (N,12,1,length) matrix and a (N,6) matrix in the CODEfull_data.hdf5 and CODEfull_labels.hdf5, respectively. N is the number of ECG recordings, length is the length of the recordings.
* Then, you can use
```nohup torchrun --nproc_per_node=2 main.py --mode 'pretrain' --model_config 'base'> pretrain_ecg_code.log 2>&1 & ``` in the command window to pre-train a backbone model for yourself. If you have more than 2 GPU, please modify the parameter 'nproc_per_node'.
# FastECG Fine-Tuning and Evaluation on the Downsteam Datasets.
* 
