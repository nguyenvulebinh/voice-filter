#  Copyright 2022 Thai-Binh Nguyen
#  Licensed under the Apache License, Version 2.0 (the "License")
import random 
from typing import Union, Tuple

from torch import nn
from transformers import PreTrainedModel
from src.model.configuration_voicefilter import VoiceFilterConfig
from src.model.enh_s2t_task import EnhancementTask

import torch
from transformers.utils import ModelOutput
from typing import Optional
import argparse
import math
from src.utils.signal_processing import normalize
from src.net.xvector_sincnet import XVectorSincNet


class VoiceFilterOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    output_text: Optional[Tuple[torch.FloatTensor]] = None
    output_speech: Optional[Tuple[torch.FloatTensor]] = None
    

class VoiceFilter(PreTrainedModel):
    config_class = VoiceFilterConfig
    base_model_prefix = "voicefilter"

    def __init__(self, config):
        super().__init__(config)
        args = argparse.Namespace(**config.enh_args)
        self.enh_model = EnhancementTask.build_model(args)
        self.enh_model._requires_grad = True
        self.wav_chunk_size = config.enh_chunk_size * config.sample_rate
        self.filter_condition_transform = nn.Linear(args.xvector_emb_dim * 2, args.xvector_emb_dim)
        self.xvector_model = XVectorSincNet()
        self.dropout = nn.Dropout(0.3)

    def load_xvector_sincnet_model(self, model_file):
        meta = torch.load(model_file, map_location='cpu')['state_dict']
        print('load_xvector_sincnet_model', self.xvector_model.load_state_dict(meta, strict=False))
        self.xvector_model = self.xvector_model.eval()
        for param in self.xvector_model.parameters():
            param.requires_grad = False
    
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
    
    def forward(self, speech, speech_lengths, 
                target_speech=None, target_spk_embedding=None):
                # model forward
        loss_enh = None

        # enh forward
        speech_chunk, target_spk_embedding, sample_num_chunks, target_speech_chunk = self.chunk_split(speech, speech_lengths, target_spk_embedding, target_speech)
        
        speech_chunk_spk_embedding = self.xvector_model(speech_chunk.unsqueeze(1))
        speaker_embed = torch.concat([target_spk_embedding, speech_chunk_spk_embedding], dim=-1)
        speaker_embed = self.filter_condition_transform(self.dropout(speaker_embed))

        speech_chunk_lengths =  torch.ones(speech_chunk.shape[0]).int().fill_(speech_chunk.shape[1])
        speech_pre_chunk, feature_mix, feature_pre, others = self.enh_model.forward_enhance(
            speech_mix=speech_chunk,
            speaker_embed=speaker_embed,
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
        return VoiceFilterOutput(
            loss=loss_enh,
            output_speech=speech_pre
        )