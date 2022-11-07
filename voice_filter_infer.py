# from src.spnn.models.asr_voice_filter.modeling_enh_asr import ASRVoiceFilter
from src.model.modeling_enh_asr import ASRVoiceFilter
import torch
from src.net.xvector_sincnet import XVectorSincNet
import os
import glob
import csv
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf

# os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
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
    pretrained_path = './model-bin/voice_enhancing'
    asr_enh_model = ASRVoiceFilter.from_pretrained(pretrained_path)
    xvector_model = load_xvector_sincnet_model('./model-bin/speaker_embedding/xvector_sincnet.pt')
    
    if use_gpu:
        asr_enh_model = asr_enh_model.cuda()
        xvector_model = xvector_model.cuda()

    print(xvector_model)
    print(asr_enh_model)
    
    mix_wav_path = "binh_and_linh_and_music.wav"
    ref_wav_path = "binh_ref_long.wav"
    output_wav_path = "binh_and_linh_and_music-binh_output_big.wav"
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
    est_wav = asr_enh_model.do_enh(mixed_wav_tf, xvector).cpu().detach().numpy()

    # write output file
    max_amp = np.abs(est_wav).max()
    mix_scaling = 1 / max_amp
    est_wav = mix_scaling * est_wav
    
    # est_wav = mixed_wav - est_wav
    sf.write(output_wav_path, est_wav, 16000)