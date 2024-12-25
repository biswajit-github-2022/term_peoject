

from ._explorers import DiffusionExplorer


@DiffusionExplorer
def explorer(launcher):
    launcher.slurm_(gpus=4, partition='learnfair')

    launcher.bind_({'solver': 'diffusion/default',
                    'dset': 'internal/music_10k_32khz'})

    with launcher.job_array():
        launcher({'filter.use': True, 'filter.idx_band': 0, "processor.use": False, 'processor.power_std': 0.4})
        launcher({'filter.use': True, 'filter.idx_band': 1, "processor.use": False, 'processor.power_std': 0.4})
        launcher({'filter.use': True, 'filter.idx_band': 2, "processor.use": True, 'processor.power_std': 0.4})
        launcher({'filter.use': True, 'filter.idx_band': 3, "processor.use": True, 'processor.power_std': 0.75})
