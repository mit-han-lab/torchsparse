from pathlib import Path

with open(f'{Path(__file__).parent}/version.txt') as f:
    __version__ = f.read().strip().split('+')[0]
