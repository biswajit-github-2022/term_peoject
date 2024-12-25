
import typing as tp

import treetable as tt

from .._base_explorers import BaseExplorer


class LMExplorer(BaseExplorer):
    eval_metrics: tp.List[str] = []

    def stages(self) -> tp.List[str]:
        return ['train', 'valid']

    def get_grid_metrics(self):
        return [
            tt.group(
                'evaluate',
                [
                    tt.leaf('epoch', '.3f'),
                    tt.leaf('duration', '.1f'),
                    tt.leaf('ping'),
                    tt.leaf('ce', '.4f'),
                    tt.leaf('ppl', '.3f'),
                    tt.leaf('fad', '.3f'),
                    tt.leaf('kld', '.3f'),
                    tt.leaf('text_consistency', '.3f'),
                    tt.leaf('chroma_cosine', '.3f'),
                ],
                align='>',
            ),
        ]
