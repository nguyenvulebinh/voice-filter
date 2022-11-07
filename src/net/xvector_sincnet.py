from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from asteroid_filterbanks import Encoder, ParamSincFB

def merge_dict(defaults: dict, custom: dict = None):
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params

class StatsPool(nn.Module):
    """Statistics pooling
    Compute temporal mean and (unbiased) standard deviation
    and returns their concatenation.
    Reference
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    """

    def forward(
        self, sequences: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass
        Parameters
        ----------
        sequences : (batch, channel, frames) torch.Tensor
            Sequences.
        weights : (batch, frames) torch.Tensor, optional
            When provided, compute weighted mean and standard deviation.
        Returns
        -------
        output : (batch, 2 * channel) torch.Tensor
            Concatenation of mean and (unbiased) standard deviation.
        """

        if weights is None:
            mean = sequences.mean(dim=2)
            std = sequences.std(dim=2, unbiased=True)

        else:
            weights = weights.unsqueeze(dim=1)
            # (batch, 1, frames)

            num_frames = sequences.shape[2]
            num_weights = weights.shape[2]
            if num_frames != num_weights:
                warnings.warn(
                    f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
                )
                weights = F.interpolate(
                    weights, size=num_frames, mode="linear", align_corners=False
                )

            v1 = weights.sum(dim=2)
            mean = torch.sum(sequences * weights, dim=2) / v1

            dx2 = torch.square(sequences - mean.unsqueeze(2))
            v2 = torch.square(weights).sum(dim=2)

            var = torch.sum(dx2 * weights, dim=2) / (v1 - v2 / v1)
            std = torch.sqrt(var)

        return torch.cat([mean, std], dim=1)

class SincNet(nn.Module):
    def __init__(self, sample_rate: int = 16000, stride: int = 1):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("PyanNet only supports 16kHz audio for now.")
            # TODO: add support for other sample rate. it should be enough to multiply
            # kernel_size by (sample_rate / 16000). but this needs to be double-checked.

        self.stride = stride

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=self.stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            )
        )
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward
        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """

        outputs = self.wav_norm1d(waveforms)

        for c, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):

            outputs = conv1d(outputs)

            # https://github.com/mravanelli/SincNet/issues/4
            if c == 0:
                outputs = torch.abs(outputs)

            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs

class XVectorSincNet(nn.Module):

    SINCNET_DEFAULTS = {"stride": 10}

    def __init__(
        self,
        sample_rate: int = 16000,
        # num_channels: int = 1,
        sincnet: dict = dict(
            stride=10,
            sample_rate=16000
        ),
        dimension: int = 512,
        # task: Optional[Task] = None,
    ):
        super(XVectorSincNet, self).__init__()

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        # self.save_hyperparameters("sincnet", "dimension")

        self.sincnet = SincNet(**sincnet)
        in_channel = 60

        self.tdnns = nn.ModuleList()
        out_channels = [512, 512, 512, 512, 1500]
        kernel_sizes = [5, 3, 3, 1, 1]
        dilations = [1, 2, 3, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, kernel_sizes, dilations
        ):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(out_channel),
                ]
            )
            in_channel = out_channel

        self.stats_pool = StatsPool()

        self.embedding = nn.Linear(in_channel * 2, dimension)

    def forward(
        self, waveforms: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        """

        outputs = self.sincnet(waveforms).squeeze(dim=1)
        for tdnn in self.tdnns:
            outputs = tdnn(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        return self.embedding(outputs)