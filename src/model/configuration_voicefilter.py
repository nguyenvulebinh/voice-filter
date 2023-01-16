#  Copyright 2022 Thai-Binh Nguyen
#  Licensed under the Apache License, Version 2.0 (the "License")

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
import yaml

logger = logging.get_logger(__name__)


class VoiceFilterConfig(PretrainedConfig):
    model_type = "voicefilter"

    def __init__(
            self,
            sample_rate=8000,
            audio_max_lengh=15,
            enh_config_yaml_file=None,
            enh_args=None,            
            enh_chunk_size= 5, # seconds
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
        self.enh_chunk_size=enh_chunk_size