"""Cursor-deterministic sample dropout; no edits to upstream/cache implementation."""
import hashlib
import random
from pathlib import Path
import torch


def source_identity():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('action_dropout_20261012.py','batched_sft_adrop_20261012.py')}


def action_positions(example):
    ids = example['input_ids']
    # Context 1280; N retained history frames each dynamic80+action52; target80.
    n, rem = divmod(ids.shape[1]-1280-80, 132)
    if ids.shape[0] != 1 or rem or not 1 <= n <= 7:
        raise ValueError('Unexpected unpadded RoboCasa SFT layout')
    positions = torch.cat([torch.arange(1280+i*132+80,1280+(i+1)*132,
                                        device=ids.device) for i in range(n)])
    return positions


def apply_action_dropout(example, cursor, seed, probability=0.25):
    if not 0 <= probability <= 1:
        raise ValueError('Invalid dropout probability')
    dropped = random.Random(seed+cursor*130363+60261012).random() < probability
    if not dropped:
        return example, False
    positions = action_positions(example)
    # Position-based replacement leaves target/labels/mask/positions intact.
    changed = dict(example)
    changed['input_ids'] = example['input_ids'].clone()
    changed['input_ids'][:,positions] = 9006
    return changed, True
