
from ..musicgen._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=64, partition=partitions)
    launcher.bind_(solver='audiogen/audiogen_base_16khz')
        launcher.bind_(dset='internal/sounds_16khz')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}

    launcher.bind_(fsdp)
    launcher(medium)
