import io
import logging
import re
import sys
import typing as tp

import julius
import torch
import torchaudio

logger = logging.getLogger(__name__)


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
                                wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
                                wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
                                wav = wav[..., :channels, :]
    else:
                raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    Utility function to clip the audio with logging if specified.

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    Convert audio to float 32 bits PCM format.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float32 PCM format

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float16 PCM format

    Args:
        wav (torch.Tensor): Input wav tensor.
        sr (int): Sampling rate.
        target_format (str): Compression format (e.g., 'mp3').
        bitrate (str): Bitrate for compression.

    Returns:
        Tuple of compressed WAV tensor and sampling rate.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
