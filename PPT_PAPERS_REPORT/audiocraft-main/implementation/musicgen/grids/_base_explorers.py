
from abc import ABC, abstractmethod
import time
import typing as tp
from dora import Explorer
import treetable as tt


def get_sheep_ping(sheep) -> tp.Optional[str]:
    ping = None
    if sheep.log is not None and sheep.log.exists():
        delta = time.time() - sheep.log.stat().st_mtime
        if delta > 3600 * 24:
            ping = f'{delta / (3600 * 24):.1f}d'
        elif delta > 3600:
            ping = f'{delta / (3600):.1f}h'
        elif delta > 60:
            ping = f'{delta / 60:.1f}m'
        else:
            ping = f'{delta:.1f}s'
    return ping


class BaseExplorer(ABC, Explorer):
    def stages(self):
        return ["train", "valid", "evaluate"]

    def get_grid_meta(self):
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
            tt.leaf("sid", align="<"),
        ]

    @abstractmethod
    def get_grid_metrics(self):
        ...

    def process_sheep(self, sheep, history):
        train = {
            "epoch": len(history),
        }
        parts = {"train": train}
        for metrics in history:
            for key, sub in metrics.items():
                part = parts.get(key, {})
                if 'duration' in sub:
                                        sub['duration'] = sub['duration'] / 60.
                part.update(sub)
                parts[key] = part
        ping = get_sheep_ping(sheep)
        if ping is not None:
            for name in self.stages():
                if name not in parts:
                    parts[name] = {}
                                parts[name]['ping'] = ping
        return parts
