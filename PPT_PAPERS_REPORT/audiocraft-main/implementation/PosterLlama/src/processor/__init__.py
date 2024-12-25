
from src.processor.base_processor import BaseProcessor
from src.processor.blip_processors import (
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor,
    BlipCaptionProcessor,
)


__all__ = [
    "BaseProcessor",
    "Blip2ImageTrainProcessor",
    "Blip2ImageEvalProcessor",
    "BlipCaptionProcessor",
]


def load_processor(name, cfg=None):
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
