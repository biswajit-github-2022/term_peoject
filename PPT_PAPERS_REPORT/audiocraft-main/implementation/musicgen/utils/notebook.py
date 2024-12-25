
try:
    import IPython.display as ipd  except ImportError:
        pass


import torch


def display_audio(samples: torch.Tensor, sample_rate: int):
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for audio in samples:
        ipd.display(ipd.Audio(audio, rate=sample_rate))
