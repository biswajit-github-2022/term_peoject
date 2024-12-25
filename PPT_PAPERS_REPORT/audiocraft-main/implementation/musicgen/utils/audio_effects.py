

import inspect
import random
import typing as tp
from functools import partial

import julius
import omegaconf
import torch
from julius import fft_conv1d, resample_frac

from ..data.audio_utils import get_aac, get_mp3

if tp.TYPE_CHECKING:
    from ..models.encodec import CompressionModel


def select_audio_effects(
    audio_effects: tp.Dict,
    weights: tp.Optional[tp.Dict] = None,
    mode: str = "all",
    max_length: tp.Optional[int] = None,
):
    if mode == "all":          out = audio_effects
    elif mode == "weighted":
                assert weights is not None
        out = {
            name: value
            for name, value in audio_effects.items()
            if random.random() < weights.get(name, 1.0)
        }
    else:
        raise ValueError(f"Unknown mode {mode}")
    if max_length is not None:
                random_keys = random.sample(list(out.keys()), max_length)
        out = {key: out[key] for key in random_keys}
    if len(out) == 0:          out = {"identity": AudioEffects.identity}
    return out


def get_audio_effects(cfg: omegaconf.DictConfig):
    assert hasattr(cfg, "audio_effects")
    cfg_audio_effects = dict(cfg["audio_effects"])
    return {
        name: partial(value, **cfg_audio_effects.get(name, {}))
        for name, value in inspect.getmembers(AudioEffects)
        if inspect.isfunction(value)
    }


def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
        pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise


def compress_with_encodec(
    tensor: torch.Tensor,
    n_q: int,
    model: "CompressionModel",
    sample_rate: int,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    model.to(tensor.device)
    model.set_num_codebooks(n_q)
    codes, scale = model.encode(
        julius.resample_frac(tensor, old_sr=sample_rate, new_sr=model.sample_rate)
    )
    compressed = model.decode(codes=codes, scale=scale)
    return audio_effect_return(
        tensor=julius.resample_frac(
            compressed, old_sr=model.sample_rate, new_sr=sample_rate
        ),
        mask=mask,
    )


def apply_compression_skip_grad(tensor: torch.Tensor, compression_fn, **kwargs):
    compressed = compression_fn(tensor.detach(), **kwargs)

        compressed = compressed[:, :, : tensor.size(-1)]

        out = tensor + (compressed - tensor).detach()

        if out.requires_grad:
        assert (
            out.grad_fn
        ), "The computation graph might be broken due to compression augmentation."

    return out


class AudioEffects:
    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.5, 1.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        speed = torch.FloatTensor(1).uniform_(*speed_range)
        new_sr = int(sample_rate * 1 / speed)
        resampled_tensor = julius.resample.resample_frac(tensor, sample_rate, new_sr)
        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        orig_shape = tensor.shape
                tensor = resample_frac(tensor, sample_rate, intermediate_freq)
                tensor = resample_frac(tensor, intermediate_freq, sample_rate)

        assert tensor.shape == orig_shape
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

                        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

                impulse_response[0] = 1.0  
        impulse_response[
            int(sample_rate * duration) - 1
        ] = volume  
                impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

                reverbed_signal = fft_conv1d(tensor, impulse_response)

                reverbed_signal = (
            reverbed_signal
            / torch.max(torch.abs(reverbed_signal))
            * torch.max(torch.abs(tensor))
        )

                tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)

    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        noise = noise.to(waveform.device)
                noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(
            tensor=julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
            ),
            mask=mask,
        )

    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
                kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)

        smoothed = fft_conv1d(tensor, kernel)
                tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)

    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def mp3_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = apply_compression_skip_grad(
            tensor, get_mp3, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def aac_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        lowpass_freq: tp.Optional[int] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = apply_compression_skip_grad(
            tensor, get_aac, sr=sample_rate, bitrate=bitrate, lowpass_freq=lowpass_freq
        )
        return audio_effect_return(tensor=out, mask=mask)
