

from contextlib import contextmanager
import typing as tp
from torch import nn
import torch


State = tp.Dict[str, torch.Tensor]


class StreamingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State = {}
        self._is_streaming = False

    def _apply_named_streaming(self, fn: tp.Any):
        for name, module in self.named_modules():
            if isinstance(module, StreamingModule):
                fn(name, module)

    def _set_streaming(self, streaming: bool):
        def _set_streaming(name, module):
            module._is_streaming = streaming
        self._apply_named_streaming(_set_streaming)

    @contextmanager
    def streaming(self):
        def _reset(name: str, module: StreamingModule):
            module._streaming_state.clear()

        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> State:
        state = dict(state)

        def _set(name: str, module: StreamingModule):
            if name:
                name += "."
            module._streaming_state.clear()
            for key, value in list(state.items()):
                                if key.startswith(name):
                    local_key = key[len(name):]
                    if '.' not in local_key:
                        module._streaming_state[local_key] = value
                        del state[key]

        self._apply_named_streaming(_set)
        assert len(state) == 0, list(state.keys())

    def flush(self, x: tp.Optional[torch.Tensor] = None):
        if x is None:
            return None
        else:
            return self(x)


class StreamingSequential(StreamingModule, nn.Sequential):
    def flush(self, x: tp.Optional[torch.Tensor] = None):
        for module in self:
            if isinstance(module, StreamingModule):
                x = module.flush(x)
            elif x is not None:
                x = module(x)
        return x
