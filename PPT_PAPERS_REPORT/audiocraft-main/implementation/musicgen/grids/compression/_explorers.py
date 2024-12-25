
import treetable as tt

from .._base_explorers import BaseExplorer


class CompressionExplorer(BaseExplorer):
    eval_metrics = ["sisnr", "visqol"]

    def stages(self):
        return ["train", "valid", "evaluate"]

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
                    tt.leaf("bandwidth", ".2f"),
                    tt.leaf("adv", ".4f"),
                    tt.leaf("d_loss", ".4f"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("bandwidth", ".2f"),
                    tt.leaf("adv", ".4f"),
                    tt.leaf("msspec", ".4f"),
                    tt.leaf("sisnr", ".2f"),
                ],
                align=">",
            ),
            tt.group(
                "evaluate", [tt.leaf(name, ".3f") for name in self.eval_metrics], align=">"
            ),
        ]
