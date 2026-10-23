"""Validated configuration; loading it does not start GPU work."""
from dataclasses import dataclass
import json
from pathlib import Path
from .experiment import PROFILES, validate_development_config

@dataclass(frozen=True)
class Settings:
    run: dict
    paths: dict
    inference: dict

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        result = cls(**data)
        run, inf = result.run, result.inference
        if run['DEV_MODE'] not in {'smoke', 'compare', 'confirm', 'full'}:
            raise ValueError('Unknown DEV_MODE')
        for profile in [run['SUBMISSION_PROFILE'], run['DEV_FULL_PROFILE'], *run['DEV_PROFILES']]:
            if profile not in PROFILES:
                raise ValueError(f'Unknown profile: {profile}')
        validate_development_config(dict(profile=run['DEV_FULL_PROFILE'],
            minutes=run['DEV_MINUTES_PER_ARM'], active_streams=run['DEV_ACTIVE_STREAMS'],
            passes=run['DEV_REPEATS']))
        if inf['MAXREQ'] < 10 or inf['CUDAGRAPH_MAXBS'] < inf['MAXREQ']:
            raise ValueError('Server and CUDA Graph must cover the 10-stream baseline')
        if inf['CHUNK'] <= 0 or inf['CHUNK'] % 64:
            raise ValueError('CHUNK must be a positive multiple of 64')
        if not 0 < inf['MEMFRAC'] < 1:
            raise ValueError('MEMFRAC must lie between 0 and 1')
        return result

    def check_inputs(self):
        for key in ('model', 'draft_model', 'wheelhouse', 'source_bundle'):
            if not Path(self.paths[key]).is_dir():
                raise FileNotFoundError(f'{key}: {self.paths[key]}')
