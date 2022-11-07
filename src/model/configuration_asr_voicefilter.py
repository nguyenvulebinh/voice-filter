#  Copyright 2022 Thai-Binh Nguyen
#  Licensed under the Apache License, Version 2.0 (the "License")

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
import yaml

logger = logging.get_logger(__name__)


class ASRVoiceFilterConfig(PretrainedConfig):
    model_type = "asr_voicefilter"

    def __init__(
            self,
            d_input=80,
            sample_rate=8000,
            n_fft=512,
            audio_max_lengh=15,            
            ignore_token_id=-1,
            enh_config_yaml_file=None,
            enh_args=None,            
            enh_chunk_size= 5, # seconds
            do_enh=True,
            do_asr=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        if enh_config_yaml_file is not None:
            with open(enh_config_yaml_file, 'r', encoding='utf-8') as f:
                self.enh_args = yaml.safe_load(f)
        else:
            self.enh_args = enh_args
        self.sample_rate = sample_rate
        self.audio_max_lengh = audio_max_lengh
        self.n_fft = n_fft
        self.n_mels = d_input
        self.enh_chunk_size=enh_chunk_size
        self.do_asr = do_asr
        self.do_enh = do_enh
        self.ignore_token_id=ignore_token_id