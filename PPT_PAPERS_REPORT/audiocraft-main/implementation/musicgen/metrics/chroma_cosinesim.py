
import torch
import torchmetrics

from ..data.audio_utils import convert_audio
from ..modules.chroma import ChromaExtractor


class ChromaCosineSimilarityMetric(torchmetrics.Metric):
    def __init__(self, sample_rate: int, n_chroma: int, radix2_exp: int, argmax: bool, eps: float = 1e-8):
        super().__init__()
        self.chroma_sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.eps = eps
        self.chroma_extractor = ChromaExtractor(sample_rate=self.chroma_sample_rate, n_chroma=self.n_chroma,
                                                radix2_exp=radix2_exp, argmax=argmax)
        self.add_state("cosine_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        assert self.weight.item() > 0, "Unable to compute with total number of comparisons <= 0"          return (self.cosine_sum / self.weight).item()  