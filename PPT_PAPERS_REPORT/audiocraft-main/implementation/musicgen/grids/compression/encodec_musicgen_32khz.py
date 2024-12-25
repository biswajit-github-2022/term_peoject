

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=8, partition=partitions)
            launcher.bind_(solver='compression/encodec_musicgen_32khz')
        launcher.bind_(dset='internal/music_400k_32khz')
        launcher()
    launcher({
        'metrics.visqol.bin': '/data/home/jadecopet/local/usr/opt/visqol',
        'label': 'visqol',
        'evaluate.metrics.visqol': True
    })
