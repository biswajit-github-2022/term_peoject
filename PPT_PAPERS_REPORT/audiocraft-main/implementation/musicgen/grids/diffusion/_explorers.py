
import treetable as tt

from .._base_explorers import BaseExplorer


class DiffusionExplorer(BaseExplorer):
    eval_metrics = ["sisnr", "visqol"]

    def stages(self):
        return ["train", "valid", "valid_ema", "evaluate", "evaluate_ema"]

    def get_grid_meta(self):
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
        ]

    def get_grid_metrics(self):
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("loss", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("loss", ".3%"),
                                    ],
                align=">",
            ),
            tt.group(
                "valid_ema",
                [
                    tt.leaf("loss", ".3%"),
                                    ],
                align=">",
            ),
            tt.group(
                "evaluate", [tt.leaf("rvm", ".4f"), tt.leaf("rvm_0", ".4f"),
                             tt.leaf("rvm_1", ".4f"), tt.leaf("rvm_2", ".4f"),
                             tt.leaf("rvm_3", ".4f"), ], align=">"
            ),
            tt.group(
                "evaluate_ema", [tt.leaf("rvm", ".4f"), tt.leaf("rvm_0", ".4f"),
                                 tt.leaf("rvm_1", ".4f"), tt.leaf("rvm_2", ".4f"),
                                 tt.leaf("rvm_3", ".4f")], align=">"
            ),
        ]
