import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from src.model.modeling_enh import VoiceFilter
import torch
from huggingface_hub import hf_hub_download
import glob
import csv
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf
from typing import Union
import io
import torchaudio

from fastapi import FastAPI, UploadFile, File, Body, Response

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

app = FastAPI()
repo_id = 'nguyenvulebinh/voice-filter'
enh_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
if use_gpu:
    enh_model = enh_model.cuda()
    
    
@app.post("/voice-filter")
async def voice_filter(raw_audio_file: UploadFile = File(...), 
                       target_speaker_file: UploadFile = File(...),):
    """
    Endpoint to process an audio file with given JSON payload.
    """
    # Load the audio file into a numpy array
    with io.BytesIO(await raw_audio_file.read()) as audio_buffer:
        raw_audio_data, samplerate = sf.read(audio_buffer, dtype='float32')
        print('recieved raw audio {}: {}s'.format(raw_audio_data.shape, len(raw_audio_data)/samplerate))
        if len(raw_audio_data.shape) > 1:
            raw_audio_data = raw_audio_data[:,0]
        if samplerate != 16000:
            return {
                "error": "sample rate must be 16000"
            }
    
    with io.BytesIO(await target_speaker_file.read()) as audio_buffer:
        target_audio_data, samplerate = sf.read(audio_buffer, dtype='float32')
        print('recieved target audio {}: {}s'.format(target_audio_data.shape, len(target_audio_data)/samplerate))
        if len(target_audio_data.shape) > 1:
            target_audio_data = target_audio_data[:,0]
        if samplerate != 16000:
            return {
                "error": "sample rate must be 16000"
            }
    xvector = cal_xvector_sincnet_embedding(enh_model.xvector_model, target_audio_data)
    print('xvector: {}'.format(xvector.shape))
    
    # Speech enhancing
    mixed_wav = raw_audio_data
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
    print('output audio {}: {}s'.format(est_wav.shape, len(est_wav)/samplerate))
    with io.BytesIO() as wav_bytes:
        sf.write(wav_bytes, est_wav, samplerate, format='WAV', subtype='PCM_16')
        wav_bytes.seek(0)
        # Return the processed audio file as a response
        return Response(content=wav_bytes.getvalue(), media_type="audio/wav")

# post an audio file to the server and get an vector representation of the audio
@app.post("/audio-embedding")
async def audio_embedding(audio_file: UploadFile = File(...),):
    # Load the audio file into a numpy array
    with io.BytesIO(await audio_file.read()) as audio_buffer:
        data, samplerate = sf.read(audio_buffer, dtype='float32')
    print('recieved audio {}: {}s'.format(data.shape, len(data)/samplerate))
    if samplerate != 16000:
        return {
            "error": "sample rate must be 16000"
        }
    xvector = cal_xvector_sincnet_embedding(enh_model.xvector_model, data)
    return {
        "embedding": xvector.tolist()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)