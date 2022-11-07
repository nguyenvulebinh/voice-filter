#  Copyright 2022 Thai-Binh Nguyen
#  Licensed under the Apache License, Version 2.0 (the "License")
import random 
from typing import Union, Tuple

from torch import nn
from transformers import PreTrainedModel
from .configuration_asr_voicefilter import ASRVoiceFilterConfig
from .enh_s2t_task import EnhancementTask

import torch
from transformers.utils import ModelOutput
from typing import Optional
import argparse
import math
from src.utils.signal_processing import normalize


class ASRVoiceFilterOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    output_text: Optional[Tuple[torch.FloatTensor]] = None
    output_speech: Optional[Tuple[torch.FloatTensor]] = None
    

class ASRVoiceFilter(PreTrainedModel):
    config_class = ASRVoiceFilterConfig
    base_model_prefix = "asr_voicefilter"

    def __init__(self, config):
        super().__init__(config)
        args = argparse.Namespace(**config.enh_args)
        self.enh_model = EnhancementTask.build_model(args)
        self.enh_model._requires_grad = True
        self.wav_chunk_size = config.enh_chunk_size * config.sample_rate
    
    def chunk_split(self, speech, speech_lengths, target_spk_embedding, target_speech=None):
        assert speech.size(-1) % self.wav_chunk_size == 0
        sample_num_chunks = [math.ceil(item / self.wav_chunk_size) for item in speech_lengths]
        speech_chunk = torch.concat([item[:num*self.wav_chunk_size].view(-1, self.wav_chunk_size) for num, item in zip(sample_num_chunks, speech)])
        target_spk_embedding = torch.concat([item.repeat(num, 1) for num, item in zip(sample_num_chunks, target_spk_embedding)])
        if target_speech is not None:
            target_speech_chunk = torch.concat([item[:num*self.wav_chunk_size].view(-1, self.wav_chunk_size) for num, item in zip(sample_num_chunks, target_speech)])
            return speech_chunk, target_spk_embedding, sample_num_chunks, target_speech_chunk
        return speech_chunk, target_spk_embedding, sample_num_chunks, None

    def merge_chunk(self, speech_pre_chunk, sample_num_chunks, input_max_length, speech_lengths):
        speech_pre = []
        idx_start = 0
        for num, len in zip(sample_num_chunks, speech_lengths):
            wav_out = torch.cat([
                speech_pre_chunk[idx_start: idx_start + num].view(-1),
                speech_pre_chunk.new([0] * (input_max_length - num * self.wav_chunk_size))
            ])
            wav_out[len:] = 0
            speech_pre.append(wav_out)
            idx_start += num
        return torch.stack(speech_pre)
    
    def do_enh(self, speech, speaker_embedding):
        # speech = normalize(speech, lengths=torch.tensor([len(speech)])).squeeze()
        speech_lengths = torch.tensor(len(speech))
        max_len = math.ceil(len(speech)/self.wav_chunk_size) * self.wav_chunk_size
        speech = torch.cat([speech, speech.new([0] * (max_len - len(speech)))])
        
        enh_speech = self.forward(
            speech=speech.unsqueeze(0), 
            speech_lengths=speech_lengths.unsqueeze(0),
            target_spk_embedding=speaker_embedding.unsqueeze(0)
        ).output_speech[0][:speech_lengths].squeeze()
        return enh_speech
    
    def forward(self, speech, speech_lengths, target_speech=None, 
                target_spk_embedding=None, 
                text=None, text_input=None, text_ctc=None):
                # model forward
        
        loss_enh = None
        if self.enh_model is not None:
            # # enh forward
            speech_chunk, target_spk_embedding, sample_num_chunks, target_speech_chunk = self.chunk_split(speech, speech_lengths, target_spk_embedding, target_speech)
            speech_chunk_lengths =  torch.ones(speech_chunk.shape[0]).int().fill_(speech_chunk.shape[1])
            speech_pre_chunk, feature_mix, feature_pre, others = self.enh_model.forward_enhance(
                speech_mix=speech_chunk,
                speaker_embed=target_spk_embedding,
                speech_lengths=speech_chunk_lengths
            )
            speech_pre = self.merge_chunk(speech_pre_chunk[0], sample_num_chunks, speech.size(-1), speech_lengths)
            
            
            if target_speech is not None:                
                loss_enh, _, _ = self.enh_model.forward_loss(
                    [speech_pre],
                    speech_lengths,
                    None,
                    None,
                    others,
                    target_speech
                ) 
                    
        # Only do speech enh
        return ASRVoiceFilterOutput(
            loss=loss_enh,
            output_speech=speech_pre
        )