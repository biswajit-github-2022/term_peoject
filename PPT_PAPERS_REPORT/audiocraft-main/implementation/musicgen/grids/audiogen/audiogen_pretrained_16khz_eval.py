

import os

from ..musicgen._explorers import GenerationEvalExplorer
from ...environment import AudioCraftEnvironment
from ... import train


def eval(launcher, batch_size: int = 32):
    opts = {
        'dset': 'audio/audiocaps_16khz',
        'solver/audiogen/evaluation': 'objective_eval',
        'execute_only': 'evaluate',
        '+dataset.evaluate.batch_size': batch_size,
        '+metrics.fad.tf.batch_size': 32,
    }
        metrics_opts = {
        'metrics.fad.tf.bin': '/data/home/jadecopet/local/usr/opt/google-research'
    }
    opt1 = {'generate.lm.use_sampling': True, 'generate.lm.top_k': 250, 'generate.lm.top_p': 0.}
    opt2 = {'transformer_lm.two_step_cfg': True}

    sub = launcher.bind(opts)
    sub.bind_(metrics_opts)

        sub(opt1, opt2)


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

    audiogen_base = launcher.bind(solver="audiogen/audiogen_base_16khz")
    audiogen_base.bind_({'autocast': False, 'fsdp.use': True})

    audiogen_base_medium = audiogen_base.bind({'continue_from': '//pretrained/facebook/audiogen-medium'})
    audiogen_base_medium.bind_({'model/lm/model_scale': 'medium'})
    eval(audiogen_base_medium, batch_size=128)
