"""Shared by both trainers: import paths, the loss and epoch-seeded loading."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXTERNAL = Path(__file__).resolve().parents[1]
for _p in (EXTERNAL, EXTERNAL / "reader"):  # Matthew's inference code: reader/, v731/
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def require_decoders():
    """Stop unless pydicom can decode JPEG Lossless and JPEG 2000, as predict.py ensures at inference."""
    from pydicom.pixels import get_decoder
    from pydicom.uid import JPEG2000, JPEG2000Lossless, JPEGLossless, JPEGLosslessSV1

    missing = [u.name for u in (JPEGLossless, JPEGLosslessSV1, JPEG2000Lossless, JPEG2000)
               if not get_decoder(u).is_available]
    if missing:
        sys.exit(f"pydicom has no decoder for {missing}: pip install -r external/train/requirements.txt")


def masked_bce(logit, y):
    """BCE over the known targets (NaN = unknown, masked)."""
    known = ~torch.isnan(y)
    return (F.binary_cross_entropy_with_logits(logit.float(), torch.nan_to_num(y), reduction="none")
            * known).sum() / known.sum().clamp_min(1)


def loader_settings(workers):
    return dict(num_workers=workers, pin_memory=True, persistent_workers=False, generator=torch.Generator())


def seed_epoch(settings, seed, epoch):
    """Reseed the loader before each epoch, so a resumed run samples the same as an unbroken one."""
    value = (seed + 1000003 * epoch) % (2**32)
    settings["generator"].manual_seed(value)
    if settings["num_workers"] == 0:
        np.random.seed(value)
