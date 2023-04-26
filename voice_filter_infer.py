from src.model.modeling_enh import VoiceFilter
import torch
from huggingface_hub import hf_hub_download
import os
import glob
import csv
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf

use_gpu = True
if use_gpu:
    if not torch.cuda.is_available():
        use_gpu = False
        
def cal_xvector_sincnet_embedding(xvector_model, ref_wav, max_length=5, sr=16000):
    wavs = []
    for i in range(0, len(ref_wav), max_length*sr):
        wav = ref_wav[i:i + max_length*sr]
        wav = np.concatenate([wav, np.zeros(max(0, max_length * sr - len(wav)))])
        wavs.append(wav)
    wavs = torch.from_numpy(np.stack(wavs))
    if use_gpu:
        wavs = wavs.cuda()
    embed = xvector_model(wavs.unsqueeze(1).float())
    return torch.mean(embed, dim=0).detach().cpu()


if __name__ == "__main__":
    # Load models
    repo_id = 'nguyenvulebinh/voice-filter'
    enh_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    if use_gpu:
        enh_model = enh_model.cuda()
        
    print(enh_model)