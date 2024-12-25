

from dataclasses import dataclass
from pathlib import Path
import logging
import typing as tp

import numpy as np
import soundfile
import torch
from torch.nn import functional as F

import av
import subprocess as sp

from .audio_utils import f32_pcm, normalize_audio


_av_initialized = False


def _init_av():
    global _av_initialized
    if _av_initialized:
        return
    logger = logging.getLogger('libav.mp3')
    logger.setLevel(logging.ERROR)
    _av_initialized = True


@dataclass(frozen=True)
class AudioFileInfo:
    sample_rate: int
    duration: float
    channels: int


def _av_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
    _init_av()
    with av.open(str(filepath)) as af:
        stream = af.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        duration = float(stream.duration * stream.time_base)
        channels = stream.channels
        return AudioFileInfo(sample_rate, duration, channels)


def _soundfile_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
    info = soundfile.info(filepath)
    return AudioFileInfo(info.samplerate, info.duration, info.channels)


def audio_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
        filepath = Path(filepath)
    if filepath.suffix in ['.flac', '.ogg']:                  return _soundfile_info(filepath)
    else:
        return _av_info(filepath)


def _av_read(filepath: tp.Union[str, Path], seek_time: float = 0, duration: float = -1.) -> tp.Tuple[torch.Tensor, int]:
    _init_av()
    with av.open(str(filepath)) as af:
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        num_frames = int(sr * duration) if duration >= 0 else -1
        frame_offset = int(sr * seek_time)
                        af.seek(int(max(0, (seek_time - 0.1)) / stream.time_base), stream=stream)
        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            current_offset = int(frame.rate * frame.pts * frame.time_base)
            strip = max(0, frame_offset - current_offset)
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != stream.channels:
                buf = buf.view(-1, stream.channels).t()
            buf = buf[:, strip:]
            frames.append(buf)
            length += buf.shape[1]
            if num_frames > 0 and length >= num_frames:
                break
        assert frames
                                wav = torch.cat(frames, dim=1)
        assert wav.shape[0] == stream.channels
        if num_frames > 0:
            wav = wav[:, :num_frames]
        return f32_pcm(wav), sr


def audio_read(filepath: tp.Union[str, Path], seek_time: float = 0.,
               duration: float = -1.0, pad: bool = False) -> tp.Tuple[torch.Tensor, int]:
    fp = Path(filepath)
    if fp.suffix in ['.flac', '.ogg']:                  info = _soundfile_info(filepath)
        frames = -1 if duration <= 0 else int(duration * info.sample_rate)
        frame_offset = int(seek_time * info.sample_rate)
        wav, sr = soundfile.read(filepath, start=frame_offset, frames=frames, dtype=np.float32)
        assert info.sample_rate == sr, f"Mismatch of sample rates {info.sample_rate} {sr}"
        wav = torch.from_numpy(wav).t().contiguous()
        if len(wav.shape) == 1:
            wav = torch.unsqueeze(wav, 0)
    else:
        wav, sr = _av_read(filepath, seek_time, duration)
    if pad and duration > 0:
        expected_frames = int(duration * sr)
        wav = F.pad(wav, (0, expected_frames - wav.shape[-1]))
    return wav, sr


def _piping_to_ffmpeg(out_path: tp.Union[str, Path], wav: torch.Tensor, sample_rate: int, flags: tp.List[str]):
        assert wav.dim() == 2, wav.shape
    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y', '-f', 'f32le', '-ar', str(sample_rate), '-ac', str(wav.shape[0]),
        '-i', '-'] + flags + [str(out_path)]
    input_ = f32_pcm(wav).t().detach().cpu().numpy().tobytes()
    sp.run(command, input=input_, check=True)


def audio_write(stem_name: tp.Union[str, Path],
                wav: torch.Tensor, sample_rate: int,
                format: str = 'wav', mp3_rate: int = 320, ogg_rate: tp.Optional[int] = None,
                normalize: bool = True, strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                loudness_compressor: bool = False,
                log_clipping: bool = True, make_parent_dir: bool = True,
                add_suffix: bool = True) -> Path:
    assert wav.dtype.is_floating_point, "wav is not floating point"
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = normalize_audio(wav, normalize, strategy, peak_clip_headroom_db,
                          rms_headroom_db, loudness_headroom_db, loudness_compressor,
                          log_clipping=log_clipping, sample_rate=sample_rate,
                          stem_name=str(stem_name))
    if format == 'mp3':
        suffix = '.mp3'
        flags = ['-f', 'mp3', '-c:a', 'libmp3lame', '-b:a', f'{mp3_rate}k']
    elif format == 'wav':
        suffix = '.wav'
        flags = ['-f', 'wav', '-c:a', 'pcm_s16le']
    elif format == 'ogg':
        suffix = '.ogg'
        flags = ['-f', 'ogg', '-c:a', 'libvorbis']
        if ogg_rate is not None:
            flags += ['-b:a', f'{ogg_rate}k']
    elif format == 'flac':
        suffix = '.flac'
        flags = ['-f', 'flac']
    else:
        raise RuntimeError(f"Invalid format {format}. Only wav or mp3 are supported.")
    if not add_suffix:
        suffix = ''
    path = Path(str(stem_name) + suffix)
    if make_parent_dir:
        path.parent.mkdir(exist_ok=True, parents=True)
    try:
        _piping_to_ffmpeg(path, wav, sample_rate, flags)
    except Exception:
        if path.exists():
                        path.unlink()
        raise
    return path


def get_spec(y, sr=16000, n_fft=4096, hop_length=128, dur=8) -> np.ndarray:
    import librosa
    import librosa.display

    spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db


def save_spectrograms(
    ys: tp.List[np.ndarray],
    sr: int,
    path: str,
    names: tp.List[str],
    n_fft: int = 4096,
    hop_length: int = 128,
    dur: float = 8.0,
):
    import matplotlib as mpl      import matplotlib.pyplot as plt      import librosa.display

    if not names:
        names = ["Ground Truth", "Audio Watermarked", "Watermark"]
    ys = [wav[: int(dur * sr)] for wav in ys]      assert len(names) == len(
        ys
    ), f"There are {len(ys)} wavs but {len(names)} names ({names})"

        BIGGER_SIZE = 10
    SMALLER_SIZE = 8
    linewidth = 234.8775  
    plt.rc("font", size=BIGGER_SIZE, family="serif")      plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.rc("axes", titlesize=BIGGER_SIZE)      plt.rc("axes", labelsize=BIGGER_SIZE)      plt.rc("xtick", labelsize=BIGGER_SIZE)      plt.rc("ytick", labelsize=SMALLER_SIZE)      plt.rc("legend", fontsize=BIGGER_SIZE)      plt.rc("figure", titlesize=BIGGER_SIZE)
    height = 1.6 * linewidth / 72.0
    fig, ax = plt.subplots(
        nrows=len(ys),
        ncols=1,
        sharex=True,
        figsize=(linewidth / 72.0, height),
    )
    fig.tight_layout()

    
    for i, ysi in enumerate(ys):
        spectrogram_db = get_spec(ysi, sr=sr, n_fft=n_fft, hop_length=hop_length)
        if i == 0:
            cax = fig.add_axes(
                [
                    ax[0].get_position().x1 + 0.01,                      ax[-1].get_position().y0,
                    0.02,
                    ax[0].get_position().y1 - ax[-1].get_position().y0,
                ]
            )
            fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(
                        np.min(spectrogram_db), np.max(spectrogram_db)
                    ),
                    cmap="magma",
                ),
                ax=ax,
                orientation="vertical",
                format="%+2.0f dB",
                cax=cax,
            )
        librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax[i],
        )
        ax[i].set(title=names[i])
        ax[i].yaxis.set_label_text(None)
        ax[i].label_outer()
    fig.savefig(path, bbox_inches="tight")
    plt.close()
