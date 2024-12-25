
from pathlib import Path
import typing as tp

import torch
import torchmetrics
from transformers import RobertaTokenizer  
from ..data.audio_utils import convert_audio
from ..environment import AudioCraftEnvironment
from ..utils.utils import load_clap_state_dict

try:
    import laion_clap  except ImportError:
    laion_clap = None


class TextConsistencyMetric(torchmetrics.Metric):

    This metric is similar to the MuLan Cycle Consistency from MusicLM (https://arxiv.org/pdf/2301.11325.pdf)
    or the CLAP score used in Make-An-Audio (https://arxiv.org/pdf/2301.12661v1.pdf).

    As a joint audio-text embedding model, a pretrained CLAP model can be used to quantify the
    similarity between audio-text pairs. We compute the CLAP embeddings from the text descriptions as
    well as the generated audio based on them, and define the MCC metric as the average cosine similarity
    between these embeddings.

    Model implementation & pre-trained checkpoints: https://github.com/LAION-AI/CLAP
        assert audio.size(0) == len(text), "Number of audio and text samples should match"
        assert torch.all(sample_rates == sample_rates[0].item()), "All items in batch should have the same sample rate"
        sample_rate = int(sample_rates[0].item())
                audio = convert_audio(audio, from_rate=sample_rate, to_rate=self.model_sample_rate, to_channels=1).mean(dim=1)
        audio_embeddings = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
        text_embeddings = self.model.get_text_embedding(text, tokenizer=self._tokenizer, use_tensor=True)
                cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings, dim=1, eps=1e-8)
        self.cosine_sum += cosine_sim.sum(dim=0)
        self.weight += torch.tensor(cosine_sim.size(0))

    def compute(self):
