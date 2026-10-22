"""MNIST loading, digit placement on the 56x56 canvas, training regimes and the eval subset."""

import torch
import torch.nn.functional as F
import torchvision
from common import (
    CANVAS,
    DATA_DIR,
    DIGIT,
    MAX_OFFSET,
    N_VAL,
    SPLIT_SEED,
    SUBSET_PER_CLASS,
)


def place(digits, dx, dy) -> torch.Tensor:
    """Paste 28x28 digit(s) onto a zero 56x56 canvas, top-left corner at column dx, row dy.

    The single placement function used for training, evaluation and the exported
    references. `digits` is (28, 28) or (N, 28, 28); `dx`, `dy` are ints or (N,) tensors
    with values in 0..28. Returns (56, 56) or (N, 56, 56) with the dtype of `digits`.
    """
    digits = torch.as_tensor(digits)
    single = digits.ndim == 2
    if single:
        digits = digits[None]
    if isinstance(dx, int) and isinstance(dy, int):
        # Same offset for all digits: zero padding gives the identical canvas, faster.
        assert 0 <= dx <= MAX_OFFSET and 0 <= dy <= MAX_OFFSET, "offset out of range"
        canvas = F.pad(digits, (dx, MAX_OFFSET - dx, dy, MAX_OFFSET - dy))
        return canvas[0] if single else canvas

    n, device = digits.shape[0], digits.device
    dx = torch.as_tensor(dx, device=device).long().expand(n)
    dy = torch.as_tensor(dy, device=device).long().expand(n)
    assert int(dx.min()) >= 0 and int(dx.max()) <= MAX_OFFSET, "dx out of range"
    assert int(dy.min()) >= 0 and int(dy.max()) <= MAX_OFFSET, "dy out of range"

    r = torch.arange(DIGIT, device=device)
    rows = (dy[:, None] + r)[:, :, None]  # (N, 28, 1)
    cols = (dx[:, None] + r)[:, None, :]  # (N, 1, 28)
    batch = torch.arange(n, device=device)[:, None, None]
    canvas = digits.new_zeros(n, CANVAS, CANVAS)
    canvas[batch, rows, cols] = digits
    return canvas[0] if single else canvas


def offset_range(regime: str, tl_max_offset: int = 0) -> tuple[int, int]:
    """Inclusive (min, max) offset range of a training regime, the same for dx and dy."""
    if regime == "tl":
        return 0, tl_max_offset
    if regime == "all":
        return 0, MAX_OFFSET
    raise ValueError(f"unknown regime {regime!r}")


def sample_offsets(
    n: int, regime: str, tl_max_offset: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform integer (dx, dy) per image within the regime's range (on CPU)."""
    lo, hi = offset_range(regime, tl_max_offset)
    dx = torch.randint(lo, hi + 1, (n,), generator=generator)
    dy = torch.randint(lo, hi + 1, (n,), generator=generator)
    return dx, dy


def load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw MNIST as uint8 images (N, 28, 28) and int64 labels (N,)."""
    ds = torchvision.datasets.MNIST(DATA_DIR, train=train, download=True)
    return ds.data.clone(), ds.targets.clone()


def to_float(images_u8: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 in [0, 1] (x / 255, no mean/std normalization)."""
    return images_u8.float() / 255


def train_val_split() -> tuple[torch.Tensor, torch.Tensor]:
    """Indices into the 60k training set: 55k train, 5k val (fixed seed)."""
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(60000, generator=g)
    return perm[N_VAL:].sort().values, perm[:N_VAL].sort().values


def eval_subset_indices() -> list[int]:
    """2000 test indices, 200 per class, fixed seed; in shuffled (not class) order."""
    _, labels = load_mnist(train=False)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(labels), generator=g)
    picked = []
    for c in range(10):
        picked.append(perm[labels[perm] == c][:SUBSET_PER_CLASS])
    subset = torch.cat(picked)
    subset = subset[torch.randperm(len(subset), generator=g)]
    return subset.tolist()


def fixed_offsets(n: int, regime: str, tl_max_offset: int, seed: int):
    """Deterministic per-image offsets (used for val and own-regime test accuracy)."""
    return sample_offsets(n, regime, tl_max_offset, torch.Generator().manual_seed(seed))
