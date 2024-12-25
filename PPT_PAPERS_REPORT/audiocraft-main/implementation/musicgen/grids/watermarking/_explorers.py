
import treetable as tt

from .._base_explorers import BaseExplorer


class WatermarkingMbExplorer(BaseExplorer):
    eval_metrics = ["acc", "bit_acc", "visqol", "fnr", "fpr", "sisnr"]

    def stages(self):
        return ["train", "valid", "valid_ema", "evaluate", "evaluate_ema"]

    def get_grid_meta(self):
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("sisnr", ".3%"),
                    tt.leaf("wm_detection_identity", ".3%"),
                    tt.leaf("wm_mb_identity", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("sisnr", ".3%"),
                    tt.leaf("wm_detection_identity", ".3%"),
                    tt.leaf("wm_mb_identity", ".3%"),
                                    ],
                align=">",
            ),
            tt.group(
                "evaluate",
                [
                    tt.leaf("aug_identity_acc", ".4f"),
                    tt.leaf("aug_identity_fnr", ".4f"),
                    tt.leaf("aug_identity_fpr", ".4f"),
                    tt.leaf("aug_identity_bit_acc", ".4f"),
                    tt.leaf("pesq", ".4f"),
                    tt.leaf("all_aug_acc", ".4f"),
                    tt.leaf("localization_acc_padding", ".4f"),
                ],
                align=">",
            ),
        ]


class WatermarkingExplorer(BaseExplorer):
    eval_metrics = ["acc", "visqol", "fnr", "fpr", "sisnr"]

    def stages(self):
        return ["train", "valid", "valid_ema", "evaluate", "evaluate_ema"]

    def get_grid_meta(self):
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("sisnr", ".3f"),
                    tt.leaf("wm_detection_identity"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("sisnr", ".3f"),
                    tt.leaf("wm_detection_identity"),
                                    ],
                align=">",
            ),
            tt.group(
                "evaluate",
                [
                    tt.leaf("aug_identity_acc", ".4f"),
                    tt.leaf("aug_identity_fnr", ".4f"),
                    tt.leaf("aug_identity_fpr", ".4f"),
                    tt.leaf("pesq", ".4f"),
                    tt.leaf("all_aug_acc", ".4f"),
                    tt.leaf("localization_acc_padding", ".4f"),

                ],
                align=">",
            ),
        ]
