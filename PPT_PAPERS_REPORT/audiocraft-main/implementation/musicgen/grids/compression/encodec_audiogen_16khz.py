

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=8, partition=partitions)
            launcher.bind_(solver='compression/encodec_audiogen_16khz')
        launcher.bind_(dset='internal/sounds_16khz')
        launcher()
