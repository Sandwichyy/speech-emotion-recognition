# --coding:utf-8--
import os
import csv

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

import time

import logging

device1=torch.device('cuda:0')
# device2=torch.device('cpu')

input_path = "./final-project-code-submission-Sandwichyy/WavTokenizer/AudioWAV"
# out_folder = './final-project-code-submission-Sandwichyy/WavTokenizer/tokens'
# os.system("rm -r %s"%(out_folder))
# os.system("mkdir -p %s"%(out_folder))
# ll="libritts_testclean500_large"
ll="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_testclean_epoch34"

#tmptmp=out_folder+"/"+ll

#os.system("rm -r %s"%(tmptmp))
#os.system("mkdir -p %s"%(tmptmp))
os.makedirs("./final-project-code-submission-Sandwichyy/WavTokenizer/tokens", exist_ok=True)


config_path = "./final-project-code-submission-Sandwichyy/WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./final-project-code-submission-Sandwichyy/WavTokenizer/WavTokenizer_small_600_24k_4096.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device1)

x = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".wav")]


# features_all=[]


for i in range(len(x)):

    wav, sr = torchaudio.load(x[i])
    # print("***:",x[i])
    # wav = convert_audio(wav, sr, 24000, 1)                             # (1,131040)
    bandwidth_id = torch.tensor([0])
    wav=wav.to(device1)
    print(i)

    # label
    filename = os.path.basename(x[i])
    label = filename[9:12]
    match label:
        case "ANG":
            label = 0
        case "DIS":
            label = 1
        case "FEA":
            label = 2
        case "HAP":
            label = 3
        case "NEU":
            label = 4
        case "SAD":
            label = 5

    # add to dictionary
    features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    features = features.permute(2, 0, 1)
    features = torch.squeeze(features)
    # features_all.append(features)
    
    data = {
        'features': features.cpu(),
        'label': label
    }

    save_path = f"./final-project-code-submission-Sandwichyy/WavTokenizer/tokens/token_{i}.pt"
    torch.save(data, save_path)






