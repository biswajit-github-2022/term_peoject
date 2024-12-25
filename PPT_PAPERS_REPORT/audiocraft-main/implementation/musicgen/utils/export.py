

from pathlib import Path
import typing as tp

from omegaconf import OmegaConf
import torch

from audiocraft import __version__


def export_encodec(checkpoint_path: tp.Union[Path, str], out_file: tp.Union[Path, str]):
    pkg = torch.load(checkpoint_path, 'cpu')
    new_pkg = {
        'best_state': pkg['best_state']['model'],
        'xp.cfg': OmegaConf.to_yaml(pkg['xp.cfg']),
        'version': __version__,
        'exported': True,
    }
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_file)
    return out_file


def export_pretrained_compression_model(pretrained_encodec: str, out_file: tp.Union[Path, str]):
    if Path(pretrained_encodec).exists():
        pkg = torch.load(pretrained_encodec)
        assert 'best_state' in pkg
        assert 'xp.cfg' in pkg
        assert 'version' in pkg
        assert 'exported' in pkg
    else:
        pkg = {
            'pretrained': pretrained_encodec,
            'exported': True,
            'version': __version__,
        }
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(pkg, out_file)


def export_lm(checkpoint_path: tp.Union[Path, str], out_file: tp.Union[Path, str]):
    pkg = torch.load(checkpoint_path, 'cpu')
    if pkg['fsdp_best_state']:
        best_state = pkg['fsdp_best_state']['model']
    else:
        assert pkg['best_state']
        best_state = pkg['best_state']['model']
    new_pkg = {
        'best_state': best_state,
        'xp.cfg': OmegaConf.to_yaml(pkg['xp.cfg']),
        'version': __version__,
        'exported': True,
    }

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_file)
    return out_file
