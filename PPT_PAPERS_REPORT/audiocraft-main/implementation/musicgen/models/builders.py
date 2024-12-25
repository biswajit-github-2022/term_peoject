

import typing as tp

import omegaconf
import torch

import audiocraft

from .. import quantization as qt
from ..modules.codebooks_patterns import (CoarseFirstPattern,
                                          CodebooksPatternProvider,
                                          DelayedPatternProvider,
                                          MusicLMPattern,
                                          ParallelPatternProvider,
                                          UnrolledPatternProvider)
from ..modules.conditioners import (BaseConditioner, ChromaStemConditioner,
                                    CLAPEmbeddingConditioner, ConditionFuser,
                                    ConditioningProvider, LUTConditioner,
                                    T5Conditioner, StyleConditioner)
from ..modules.diffusion_schedule import MultiBandProcessor, SampleProcessor
from ..utils.utils import dict_from_config
from .encodec import (CompressionModel, EncodecModel,
                      InterleaveStereoCompressionModel)
from .lm import LMModel
from .lm_magnet import MagnetLMModel
from .unet import DiffusionUnet
from .watermark import WMModel


def get_quantizer(
    quantizer: str, cfg: omegaconf.DictConfig, dimension: int
) -> qt.BaseQuantizer:
    klass = {"no_quant": qt.DummyQuantizer, "rvq": qt.ResidualVectorQuantizer}[
        quantizer
    ]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != "no_quant":
        kwargs["dimension"] = dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name == "seanet":
        kwargs = dict_from_config(getattr(cfg, "seanet"))
        encoder_override_kwargs = kwargs.pop("encoder")
        decoder_override_kwargs = kwargs.pop("decoder")
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        encoder = audiocraft.modules.SEANetEncoder(**encoder_kwargs)
        decoder = audiocraft.modules.SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_compression_model(cfg: omegaconf.DictConfig) -> CompressionModel:
    if cfg.lm_model in ["transformer_lm", "transformer_lm_magnet"]:
        kwargs = dict_from_config(getattr(cfg, "transformer_lm"))
        n_q = kwargs["n_q"]
        q_modeling = kwargs.pop("q_modeling", None)
        codebooks_pattern_cfg = getattr(cfg, "codebooks_pattern")
        attribute_dropout = dict_from_config(getattr(cfg, "attribute_dropout"))
        cls_free_guidance = dict_from_config(getattr(cfg, "classifier_free_guidance"))
        cfg_prob, cfg_coef = (
            cls_free_guidance["training_dropout"],
            cls_free_guidance["inference_coef"],
        )
        fuser = get_condition_fuser(cfg)
        condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)
        if len(fuser.fuse2cond["cross"]) > 0:              kwargs["cross_attention"] = True
        if codebooks_pattern_cfg.modeling is None:
            assert (
                q_modeling is not None
            ), "LM model should either have a codebook pattern defined or transformer_lm.q_modeling"
            codebooks_pattern_cfg = omegaconf.OmegaConf.create(
                {"modeling": q_modeling, "delay": {"delays": list(range(n_q))}}
            )

        pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)
        lm_class = MagnetLMModel if cfg.lm_model == "transformer_lm_magnet" else LMModel
        return lm_class(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
            **kwargs,
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected LM model {cfg.lm_model}")


def get_conditioner_provider(
    output_dim: int, cfg: omegaconf.DictConfig
) -> ConditioningProvider:
    fuser_cfg = getattr(cfg, "fuser")
    fuser_methods = ["sum", "cross", "prepend", "input_interpolate"]
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


def get_codebooks_pattern_provider(
    n_q: int, cfg: omegaconf.DictConfig
) -> CodebooksPatternProvider:
    assert sample_rate in [
        16000,
        32000,
    ], "unsupported sample rate for debug compression model"
    model_ratios = {
        16000: [10, 8, 8],          32000: [10, 8, 16],      }
    ratios: tp.List[int] = model_ratios[sample_rate]
    frame_rate = 25
    seanet_kwargs: dict = {
        "n_filters": 4,
        "n_residual_layers": 1,
        "dimension": 32,
        "ratios": ratios,
    }
    encoder = audiocraft.modules.SEANetEncoder(**seanet_kwargs)
    decoder = audiocraft.modules.SEANetDecoder(**seanet_kwargs)
    quantizer = qt.ResidualVectorQuantizer(dimension=32, bins=400, n_q=4)
    init_x = torch.randn(8, 32, 128)
    quantizer(init_x, 1)      compression_model = EncodecModel(
        encoder,
        decoder,
        quantizer,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        channels=1,
    ).to(device)
    return compression_model.eval()


def get_diffusion_model(cfg: omegaconf.DictConfig):
        channels = cfg.channels
    num_steps = cfg.schedule.num_steps
    return DiffusionUnet(chin=channels, num_steps=num_steps, **cfg.diffusion_unet)


def get_processor(cfg, sample_rate: int = 24000):
    sample_processor = SampleProcessor()
    if cfg.use:
        kw = dict(cfg)
        kw.pop("use")
        kw.pop("name")
        if cfg.name == "multi_band_processor":
            sample_processor = MultiBandProcessor(sample_rate=sample_rate, **kw)
    return sample_processor


def get_debug_lm_model(device="cpu"):
    import audioseal

    from .watermark import AudioSeal

        assert hasattr(
        cfg, "seanet"
    ), "Missing required `seanet` parameters in AudioSeal config"
    encoder, decoder = get_encodec_autoencoder("seanet", cfg)

        kwargs = (
        dict_from_config(getattr(cfg, "audioseal")) if hasattr(cfg, "audioseal") else {}
    )
    nbits = kwargs.get("nbits", 0)
    hidden_size = getattr(cfg.seanet, "dimension", 128)
    msg_processor = audioseal.MsgProcessor(nbits, hidden_size=hidden_size)

        def _get_audioseal_detector():
                seanet_cfg = dict_from_config(cfg.seanet)
        seanet_cfg.pop("encoder")
        seanet_cfg.pop("decoder")
        detector_cfg = dict_from_config(cfg.detector)

        typed_seanet_cfg = audioseal.builder.SEANetConfig(**seanet_cfg)
        typed_detector_cfg = audioseal.builder.DetectorConfig(**detector_cfg)
        _cfg = audioseal.builder.AudioSealDetectorConfig(
            nbits=nbits, seanet=typed_seanet_cfg, detector=typed_detector_cfg
        )
        return audioseal.builder.create_detector(_cfg)

    detector = _get_audioseal_detector()
    generator = audioseal.AudioSealWM(
        encoder=encoder, decoder=decoder, msg_processor=msg_processor
    )
    model = AudioSeal(generator=generator, detector=detector, nbits=nbits)

    device = torch.device(getattr(cfg, "device", "cpu"))
    dtype = getattr(torch, getattr(cfg, "dtype", "float32"))
    return model.to(device=device, dtype=dtype)
