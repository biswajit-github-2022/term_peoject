
from . import builders, musicgen


class AudioGenSolver(musicgen.MusicGenSolver):
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.SOUND
