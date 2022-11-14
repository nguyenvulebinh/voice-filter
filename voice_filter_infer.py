from src.model.modeling_enh_asr import ASRVoiceFilter
import torch
from src.net.xvector_sincnet import XVectorSincNet
from huggingface_hub import hf_hub_download
import os
import glob
import csv
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf


use_gpu = False
if use_gpu:
    if not torch.cuda.is_available():
        use_gpu = False
        
        
def load_xvector_sincnet_model(model_file):
    meta = torch.load(model_file, map_location='cpu')['state_dict']
    xvector_model = XVectorSincNet()
    print('load_xvector_sincnet_model', xvector_model.load_state_dict(meta, strict=False))
    xvector_model = xvector_model.eval()
    return xvector_model

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
    enh_model = ASRVoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    xvector_sincnet_model_path = hf_hub_download(repo_id=repo_id, filename="xvector_sincnet.pt", cache_dir='./cache')
    xvector_model = load_xvector_sincnet_model(xvector_sincnet_model_path)
    if use_gpu:
        enh_model = enh_model.cuda()
        xvector_model = xvector_model.cuda()
        
    # Load some audio sample
    mix_wav_path = hf_hub_download(repo_id=repo_id, filename="binh_linh_newspaper_music_noise.wav", cache_dir='./cache')
    # ref_wav_path = hf_hub_download(repo_id=repo_id, filename="binh_ref_long.wav", cache_dir='./cache')
    ref_wav_path = hf_hub_download(repo_id=repo_id, filename="linh_ref_long.wav", cache_dir='./cache')
    output_wav_path = "output.wav"
    mixed_wav, _ = librosa.load(mix_wav_path, sr=16000)
    ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
    
    
    # Calculate target speaker embedding
    xvector = cal_xvector_sincnet_embedding(xvector_model, ref_wav)
    # Speech enhancing
    max_amp = np.abs(mixed_wav).max()
    mix_scaling = 1 / max_amp
    mixed_wav = mix_scaling * mixed_wav
    mixed_wav_tf = torch.from_numpy(mixed_wav)
    if use_gpu:
        mixed_wav_tf = mixed_wav_tf.cuda()
        xvector= xvector.cuda()
    est_wav = enh_model.do_enh(mixed_wav_tf, xvector).cpu().detach().numpy()
    # Normalize estimated wav
    max_amp = np.abs(est_wav).max()
    mix_scaling = 1 / max_amp
    est_wav = mix_scaling * est_wav
    # write output file
    sf.write(output_wav_path, est_wav, 16000)