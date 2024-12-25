

import os

from ..musicgen._explorers import GenerationEvalExplorer
from ...environment import AudioCraftEnvironment
from ... import train


def eval(launcher, batch_size: int = 32):
    opts = {
        'dset': 'audio/musiccaps_32khz',
        'solver/musicgen/evaluation': 'objective_eval',
        'execute_only': 'evaluate',
        '+dataset.evaluate.batch_size': batch_size,
        '+metrics.fad.tf.batch_size': 16,
    }
        metrics_opts = {
        'metrics.fad.tf.bin': '/data/home/jadecopet/local/usr/opt/google-research'
    }

    sub = launcher.bind(opts)
    sub.bind_(metrics_opts)

        sub()


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
        magnet = launcher.bind(solver="magnet/magnet_32khz")

        fsdp = {'autocast': False, 'fsdp.use': True}

        segdur_10secs = {'dataset.segment_duration': 10,
                         'generate.lm.decoding_steps': [20, 10, 10, 10]}

                magnet_small_10secs = magnet.bind({'continue_from': '//pretrained/facebook/magnet-small-10secs'})
        magnet_small_10secs.bind_(segdur_10secs)
        eval(magnet_small_10secs, batch_size=128)

        magnet_medium_10secs = magnet.bind({'continue_from': '//pretrained/facebook/magnet-medium-10secs'})
        magnet_medium_10secs.bind_(segdur_10secs)
        magnet_medium_10secs.bind_({'model/lm/model_scale': 'medium'})
        magnet_medium_10secs.bind_(fsdp)
        eval(magnet_medium_10secs, batch_size=128)

                magnet_small_30secs = magnet.bind({'continue_from': '//pretrained/facebook/magnet-small-30secs'})
        eval(magnet_small_30secs, batch_size=128)

        magnet_medium_30secs = magnet.bind({'continue_from': '//pretrained/facebook/magnet-medium-30secs'})
        magnet_medium_30secs.bind_({'model/lm/model_scale': 'medium'})
        magnet_medium_30secs.bind_(fsdp)
        eval(magnet_medium_30secs, batch_size=128)
