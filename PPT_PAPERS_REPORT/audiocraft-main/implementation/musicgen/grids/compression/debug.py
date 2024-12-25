

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=2, partition=partitions)
    launcher.bind_(solver='compression/debug')

    with launcher.job_array():
                launcher()
                launcher({'rvq.bins': 2048, 'rvq.n_q': 4})
