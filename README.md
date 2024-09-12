# FastECG
# A Pytorch implementation of our paper Computation-Efficient Semi-Supervised Learning for ECG-based Cardiovascular Diseases Detection
# [Arxiv](https://arxiv.org/pdf/2406.14377)
# Preliminaries
* Four downsteam datasets: The Chapman-shaoxing dataset, the PTB-XL dataset, the Ningbo dataset, the G12EC dataset.
* If you want to retrain the backbone, please download the CODE-15% via https://zenodo.org/records/4916206
* quick download the four downsteam datasets: wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
* The pretrained base backbone (base_checkpoint.pkl). The medium and large backbones are too large to upload, we will provide the link to access them later on.
