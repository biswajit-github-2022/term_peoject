

import os

from ._explorers import GenerationEvalExplorer
from ...environment import AudioCraftEnvironment
from ... import train


def eval(launcher, batch_size: int = 32, eval_melody: bool = False):
    opts = {
        'dset': 'audio/musiccaps_32khz',
        'solver/musicgen/evaluation': 'objective_eval',
        'execute_only': 'evaluate',
        '+dataset.evaluate.batch_size': batch_size,
        '+metrics.fad.tf.batch_size': 16,
    }
        chroma_opts = {
        'dset': 'internal/music_400k_32khz',
        'dataset.evaluate.segment_duration': 30,
        'dataset.evaluate.num_samples': 1000,
        'evaluate.metrics.chroma_cosine': True,
        'evaluate.metrics.fad': False,
        'evaluate.metrics.kld': False,
        'evaluate.metrics.text_consistency': False,
    }
        metrics_opts = {
        'metrics.fad.tf.bin': '/data/home/jadecopet/local/usr/opt/google-research'
    }
    opt1 = {'generate.lm.use_sampling': True, 'generate.lm.top_k': 250, 'generate.lm.top_p': 0.}
    opt2 = {'transformer_lm.two_step_cfg': True}

    sub = launcher.bind(opts)
    sub.bind_(metrics_opts)

        sub(opt1, opt2)

    if eval_melody:
                sub(opt1, opt2, chroma_opts)


@GenerationEvalExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=4, partition=partitions)

    if 'REGEN' not in os.environ:
        folder = train.main.dora.dir / 'grids' / __name__.split('.', 2)[-1]
        with launcher.job_array():
            for sig in folder.iterdir():
                if not sig.is_symlink():
                    continue
                xp = train.main.get_xp_from_sig(sig.name)
                launcher(xp.argv)
        return

    with launcher.job_array():
        musicgen_base = launcher.bind(solver="musicgen/musicgen_base_32khz")
        musicgen_base.bind_({'autocast': False, 'fsdp.use': True})

                musicgen_base_small = musicgen_base.bind({'continue_from': '//pretrained/facebook/musicgen-small'})
        eval(musicgen_base_small, batch_size=128)

        musicgen_base_medium = musicgen_base.bind({'continue_from': '//pretrained/facebook/musicgen-medium'})
        musicgen_base_medium.bind_({'model/lm/model_scale': 'medium'})
        eval(musicgen_base_medium, batch_size=128)

        musicgen_base_large = musicgen_base.bind({'continue_from': '//pretrained/facebook/musicgen-large'})
        musicgen_base_large.bind_({'model/lm/model_scale': 'large'})
        eval(musicgen_base_large, batch_size=128)

                musicgen_melody = launcher.bind(solver="musicgen/musicgen_melody_32khz")
        musicgen_melody.bind_({'autocast': False, 'fsdp.use': True})

        musicgen_melody_medium = musicgen_melody.bind({'continue_from': '//pretrained/facebook/musicgen-melody'})
        musicgen_melody_medium.bind_({'model/lm/model_scale': 'medium'})
        eval(musicgen_melody_medium, batch_size=128, eval_melody=True)
