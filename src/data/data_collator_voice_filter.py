#  Copyright 2022 Thai-Binh Nguyen
#  Licensed under the Apache License, Version 2.0 (the "License")
import random
from typing import Dict, Union
import torch
import numpy as np
import warnings
from tqdm import tqdm
import librosa
from src.utils.signal_processing import rescale, remove_signal, pad_signal
from src.utils.signal_processing import reverberate
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import math
import os, json
import time

warnings.simplefilter("ignore", np.ComplexWarning)

# @dataclass
class VoiceFilterDataCollator:

    @staticmethod
    def mapping_speaker_to_sample(dataset):
        speaker_id = dataset['speaker_id']
        mapping = dict({})
        for idx in tqdm(range(len(speaker_id)), desc='Building speaker mapping....'):
            speaker = str(speaker_id[idx])
            if mapping.get(speaker, None) is None:
                mapping[speaker] = []
            mapping[speaker].append(idx)
        return mapping
    
    @staticmethod
    def mapping_speaker_to_embed(dataset, set_uid=None):
        speaker_id = dataset['speaker_id']
        speaker_id = []
        speaker_embed = []
        for item in tqdm(dataset, desc='Collecting speaker embedding....'):
            if item['xvector_sincnet'] is not None:
                if set_uid is None or item['speaker_id'] in set_uid:
                    speaker_id.append(item['speaker_id'])
                    speaker_embed.append(item['xvector_sincnet'])
        
        speaker_embed = np.array(speaker_embed)
        speaker_embed_similarity = cosine_similarity(speaker_embed, speaker_embed)

        mapping = dict({})
        for idx in tqdm(range(len(speaker_id)), desc='Building speaker embedding....'):
            current_speaker_id = speaker_id[idx]
            # similar_speaker_id = topk_by_partition(speaker_embed_similarity[idx], k=30, ascending=True)[0]
            similar_speaker_id = torch.topk(torch.from_numpy(speaker_embed_similarity[idx]), k=31).indices
            if mapping.get(current_speaker_id, None) is not None:
                mapping[speaker_id[idx]] = None
            else:
                mapping[speaker_id[idx]] = {
                    "embed" : speaker_embed[idx],
                    "similar_speaker": [speaker_id[i] for i in similar_speaker_id][1:]
                }
        return mapping


    def apply_reverb(self, wav, rir_raw, sample_rate=16000):
        # get rir wav (0.1 -> 0.3s)
        rir_raw = rir_raw[:int(sample_rate)//random.choice([1,2,2,4,4,4,8,8,8,8])]
        rev_waveform = reverberate(wav, rir_raw, rescale_amp="avg")
        return rev_waveform, len(rir_raw)


    def __init__(self, 
                 dataset, config, 
                 spk_embed=None,
                 noise_dataset=None, 
                 reverb_dataset=None, 
                 dereverb=True,
                 cache_uid2embed=None):
        self.dataset = dataset
        self.noise_dataset = noise_dataset
        self.reverb_dataset = reverb_dataset
        self.wav_max_length = config.audio_max_lengh * config.sample_rate
        self.wav_chunk_size = config.enh_chunk_size * config.sample_rate
        
        self.dereverb = dereverb
        self.uid2sid = self.mapping_speaker_to_sample(dataset)
        self.set_uid = set(self.uid2sid.keys())
        self.uid2embed = self.mapping_speaker_to_embed(spk_embed, set_uid = self.set_uid)
        self.list_speaker = set(self.uid2sid.keys())
        self.config = config
    
    def augment(self, audio_target, audio_other, max_len,
                random_single=0.3, 
                add_noise_ratio=0.8,
                add_reverb_ratio=0.3,
                idx=0):
        
        if self.noise_dataset is not None and random.random() < add_noise_ratio:
            noise = torch.from_numpy(self.noise_dataset[random.randint(0, len(self.noise_dataset) - 1)]['audio']['array']).float()
        else:
            noise = None

        ##############DEBUG################
        # import soundfile as sf
        # import os
        # import shutil
        # save_debug_audio_path = os.path.join('./cache/audio/', str(idx))
        # if os.path.exists(save_debug_audio_path):
        #     shutil.rmtree(save_debug_audio_path)
        # os.makedirs(save_debug_audio_path)

        # sf.write(os.path.join(save_debug_audio_path, 'target_raw.wav'), audio_target.numpy(), 16000)
        # if noise is not None:
        #     sf.write(os.path.join(save_debug_audio_path, 'noise_raw.wav'), noise.numpy(), 16000)
        # for other_raw_idx in range(len(audio_other)):
        #     sf.write(os.path.join(save_debug_audio_path, f'other_{other_raw_idx}_raw.wav'), audio_other[other_raw_idx].numpy(), 16000)
        ###################################


        # random change db level
        target_lvl = np.clip(random.normalvariate(-20.43, 15.57), -27, -17)
        audio_target = rescale(audio_target, torch.tensor(audio_target.size(-1)), target_lvl, scale="dB")
        if noise is not None:
            noise_lvl = np.clip(random.normalvariate(-20.43, 15.57), -35, -25)
            noise = rescale(noise, torch.tensor(noise.size(-1)), noise_lvl, scale="dB")
        others_lvl = [np.clip(random.normalvariate(-20.43, 15.57), -35, -17) for _ in range(len(audio_other))]
        audio_other = [rescale(item, torch.tensor(item.size(-1)), lvl, scale="dB") for item, lvl in zip(audio_other, others_lvl)]

        ##############DEBUG################
        # sf.write(os.path.join(save_debug_audio_path, 'target_norm_{:.2f}.wav'.format(target_lvl)), audio_target.numpy(), 16000)
        # if noise is not None:
        #     sf.write(os.path.join(save_debug_audio_path, 'noise_norm_{:.2f}.wav'.format(noise_lvl)), noise.numpy(), 16000)
        # for other_raw_idx in range(len(audio_other)):
        #     sf.write(os.path.join(save_debug_audio_path, 'other_{}_norm_{:.2f}.wav'.format(other_raw_idx, others_lvl[other_raw_idx])), audio_other[other_raw_idx].numpy(), 16000)
        ###################################


        # random delete signal
        audio_target = remove_signal(audio_target)
        # if noise is not None:
        #     noise = remove_signal(noise)
        audio_other = [remove_signal(item) for item in audio_other]

        ##############DEBUG################
        # sf.write(os.path.join(save_debug_audio_path, 'target_remove.wav'), audio_target.numpy(), 16000)
        # # if noise is not None:
        # #     sf.write(os.path.join(save_debug_audio_path, 'noise_remove.wav'), noise.numpy(), 16000)
        # for other_raw_idx in range(len(audio_other)):
        #     sf.write(os.path.join(save_debug_audio_path, 'other_{}_remove.wav'.format(other_raw_idx)), audio_other[other_raw_idx].numpy(), 16000)
        ###################################

        # apply reverb
        if self.reverb_dataset is not None and random.random() < add_reverb_ratio:
            reverb = torch.from_numpy(self.reverb_dataset[random.randint(0, len(self.reverb_dataset) - 1)]['audio']['array']).float()
            audio_target_reverb, target_reverb_len = self.apply_reverb(audio_target, reverb)
            if noise is not None:
                noise, noise_reverb_len = self.apply_reverb(noise, reverb)
            audio_other = [self.apply_reverb(item, reverb) for item in audio_other]
            audio_other_reverb_len = [item[1] for item in audio_other]
            audio_other = [item[0] for item in audio_other]
        else:
            audio_target_reverb = None


        ##############DEBUG################
        # if audio_target_reverb is not None:
        #     sf.write(os.path.join(save_debug_audio_path, 'target_reverb_{}.wav'.format(target_reverb_len)), audio_target_reverb.numpy(), 16000)
        #     if noise is not None:
        #         sf.write(os.path.join(save_debug_audio_path, 'noise_reverb_{}.wav'.format(noise_reverb_len)), noise.numpy(), 16000)
        #     for other_raw_idx in range(len(audio_other)):
        #         sf.write(os.path.join(save_debug_audio_path, 'other_{}_reverb_{}.wav'.format(other_raw_idx, audio_other_reverb_len[other_raw_idx])), audio_other[other_raw_idx].numpy(), 16000)
        ###################################

        # pad signal
        max_end = max([audio_target.size(-1)] + [item.size(-1) for item in audio_other] + ([noise.size(-1)] if noise is not None else []))
        max_end = min([max_end, max_len])
        audio_target = pad_signal(audio_target, max_end, max_len)
        if audio_target_reverb is not None:
            audio_target_reverb = pad_signal(audio_target_reverb, max_end, max_len)
        if noise is not None:
            noise = pad_signal(noise, max_end, max_len)
        audio_other = [pad_signal(item, max_end, max_len) for item in audio_other]

        ##############DEBUG################
        # sf.write(os.path.join(save_debug_audio_path, 'target_pad.wav'), audio_target.numpy(), 16000)
        # if noise is not None:
        #     sf.write(os.path.join(save_debug_audio_path, 'noise_pad.wav'), noise.numpy(), 16000)
        # for other_raw_idx in range(len(audio_other)):
        #     sf.write(os.path.join(save_debug_audio_path, 'other_{}_pad.wav'.format(other_raw_idx)), audio_other[other_raw_idx].numpy(), 16000)
        ###################################

        # sum signal
        mix_type = []
        single_voice = True
        if audio_target_reverb is not None:
            mix_wav = audio_target_reverb
            mix_type.append('reverb')
        else:
            mix_wav = audio_target
        if noise is not None:
            mix_wav = mix_wav + noise
            mix_type.append('noise')
        if random.random() > random_single:
            for item_audio in audio_other:
                mix_wav = mix_wav + item_audio
                single_voice = False
            mix_type.append('{}-others'.format(len(audio_other)))

        ##############DEBUG################
        # if self.dereverb or audio_target_reverb is None:
        #     sf.write(os.path.join(save_debug_audio_path, 'target_dereverb-{}.wav'.format(self.dereverb)), audio_target.numpy(), 16000)
        # else:
        #     sf.write(os.path.join(save_debug_audio_path, 'target_dereverb-{}.wav'.format(self.dereverb)), audio_target_reverb.numpy(), 16000)
        # sf.write(os.path.join(save_debug_audio_path, 'mix_{}.wav'.format('_'.join(mix_type))), mix_wav.numpy(), 16000)
        ###################################

        if self.dereverb or audio_target_reverb is None:
            return mix_wav, audio_target, max_end, single_voice
        else:
            return mix_wav, audio_target_reverb, max_end, single_voice

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        
        ##############DEBUG################
        # start_time = time.time()
        ###################################
        
        # Target speaker
        batch_speaker_id = [str(item['speaker_id']) for item in features]
        list_audio_target = [torch.from_numpy(item['audio']['array']).float() for item in features]
        speech_lengths = [item.size(-1) for item in list_audio_target]
        
        ##############DEBUG################
        # print("Get target speaker audio time: {:.2f}s".format(time.time()-start_time))
        # start_time = time.time()
        ###################################
        
        # Other speaker
        list_audio_others = []
        for target_speaker_id in batch_speaker_id:
            if self.uid2embed.get(target_speaker_id, None) is not None:
                
                ##############DEBUG################
                # start_time_item = time.time()
                ###################################
                
                num_other_speaker = random.choice([1] * 4 + [2, 3])
                negative_speaker_id = self.uid2embed[str(target_speaker_id)]['similar_speaker']
                negative_speaker_id = negative_speaker_id + random.choices(list(self.set_uid-set(negative_speaker_id)), k=30)
                
                negative_speaker_id = random.choices(negative_speaker_id, k=num_other_speaker)
                ##############DEBUG################
                # negative_speaker_id = random.choices(negative_speaker_id[:5], k=num_other_speaker)
                ###################################
                
                negative_sample_idx = [random.choice(self.uid2sid[item]) for item in negative_speaker_id]
                
                ##############DEBUG################
                # print("Get negative_speaker_id time: {:.2f}s".format(time.time()-start_time_item))
                # start_time_item = time.time()
                ################################### 
                
                negative_speaker_sample = [self.dataset[item] for item in negative_sample_idx]
                
                ##############DEBUG################
                # print("Get sample time: {:.2f}s".format(time.time()-start_time_item))
                # start_time_item = time.time()
                ################################### 
                
                current_audio_others = [torch.from_numpy(item['audio']['array']).float() for item in negative_speaker_sample]
                
                ##############DEBUG################
                # print("Get audio time: {:.2f}s".format(time.time()-start_time_item))
                # start_time_item = time.time()
                ################################### 
                
                speech_lengths.extend([item.size(-1) for item in current_audio_others])
                list_audio_others.append(current_audio_others)
                
                ##############DEBUG################
                # print("Get collect time: {:.2f}s".format(time.time()-start_time_item))
                # start_time_item = time.time()
                ################################### 
            else:
                list_audio_others.append([])
                
        ##############DEBUG################
        # print("Get {} speaker audio time: {:.2f}s".format(len(batch_speaker_id), time.time()-start_time))
        # start_time = time.time()
        ###################################

        # Mix
        batch_mix_wav = []
        batch_mix_wav_len = []
        batch_target_wav = []
        batch_target_spk_embed = []
        max_len = max(speech_lengths)
        max_len = min(math.ceil(max_len/self.wav_chunk_size) * self.wav_chunk_size, self.wav_max_length)
        for idx, (target_speaker_id, audio_target, audio_other) in enumerate(zip(batch_speaker_id, list_audio_target, list_audio_others)):
            mixed_wav, target_wav, mixed_wav_len, single_voice = self.augment(audio_target, audio_other, max_len, idx=idx)
            if mixed_wav is not None and self.uid2embed.get(target_speaker_id, None) is not None:
                batch_mix_wav.append(mixed_wav)
                batch_target_wav.append(target_wav)
                if single_voice:                
                    batch_target_spk_embed.append(np.zeros_like(self.uid2embed[target_speaker_id]['embed']))
                else:
                    batch_target_spk_embed.append(self.uid2embed[target_speaker_id]['embed'])
                batch_mix_wav_len.append(mixed_wav_len)
        
        ##############DEBUG################
        # print("Mix audio time: {:.2f}s".format(time.time()-start_time))
        # start_time = time.time()
        ###################################    

        return {
            'speech': torch.stack(batch_mix_wav).float(),
            'speech_lengths': torch.tensor(batch_mix_wav_len),
            'target_speech': torch.stack(batch_target_wav).float(),
            'target_spk_embedding': torch.tensor(batch_target_spk_embed).float()
        }
