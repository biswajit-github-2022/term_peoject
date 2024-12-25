
from ..musicgen._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=32, partition=partitions)
    launcher.bind_(solver='magnet/audio_magnet_16khz')
        launcher.bind_(dset='internal/sounds_16khz')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}

        launcher.slurm_(gpus=32).bind_(label='32gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub()

        launcher.slurm_(gpus=64).bind_(label='64gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub(medium, fsdp)
