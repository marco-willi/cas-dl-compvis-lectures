"""MLP, CNN+FC and CNN+GAP with flat module names (conv1..3, fc1, fc2).

Only Conv2d (stride 1, same padding), ReLU, MaxPool2d(2), global average pooling,
Flatten and Linear are used: the browser demo implements exactly these.
"""

import torch
import torch.nn.functional as F
from common import CANVAS
from torch import nn

CHANNELS = (16, 32, 64)  # spec default 8/16/32 gave cnn_gap_tl < 95 % (see README)
MLP_HIDDEN = 128
FC_HIDDEN = 32
N_CLASSES = 10


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(CANVAS * CANVAS, MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, N_CLASSES)

    def forward_with_intermediates(self, x: torch.Tensor):
        return self.fc2(F.relu(self.fc1(x.flatten(1)))), {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)[0]


class _CNN(nn.Module):
    """Shared trunk: 3 same-padded convs, ReLU, two 2x2 max-pools -> C3 x 14 x 14."""

    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv2d(1, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 5, padding=2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns conv1 and conv3 outputs, both after ReLU."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(F.max_pool2d(a1, 2)))
        a3 = F.relu(self.conv3(F.max_pool2d(a2, 2)))
        return a1, a3

    def head(self, a3: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_with_intermediates(self, x: torch.Tensor):
        a1, a3 = self.trunk(x)
        return self.head(a3), {"conv1": a1, "conv3": a3}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)[0]


class CNNFC(_CNN):
    def __init__(self, channels: tuple[int, int, int] = CHANNELS) -> None:
        super().__init__(channels)
        side = CANVAS // 4
        self.fc1 = nn.Linear(channels[2] * side * side, FC_HIDDEN)
        self.fc2 = nn.Linear(FC_HIDDEN, N_CLASSES)

    def head(self, a3: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(a3.flatten(1))))


class CNNGAP(_CNN):
    def __init__(self, channels: tuple[int, int, int] = CHANNELS) -> None:
        super().__init__(channels)
        self.fc1 = nn.Linear(channels[2], N_CLASSES)

    def head(self, a3: torch.Tensor) -> torch.Tensor:
        return self.fc1(a3.mean(dim=(2, 3)))


def build_model(arch: str, channels: tuple[int, int, int] = CHANNELS) -> nn.Module:
    if arch == "mlp":
        return MLP()
    if arch == "cnn_fc":
        return CNNFC(channels)
    if arch == "cnn_gap":
        return CNNGAP(channels)
    raise ValueError(f"unknown arch {arch!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_spec(arch: str, channels: tuple[int, int, int] = CHANNELS) -> list[dict]:
    """Every op in forward order, as executed literally by the browser."""

    def linear(name, n_in, n_out):
        return {"name": name, "type": "linear", "in": n_in, "out": n_out}

    if arch == "mlp":
        return [
            {"type": "flatten"},
            linear("fc1", CANVAS * CANVAS, MLP_HIDDEN),
            {"type": "relu"},
            linear("fc2", MLP_HIDDEN, N_CLASSES),
        ]
    c1, c2, c3 = channels
    trunk = [
        {"name": "conv1", "type": "conv2d", "in": 1, "out": c1, "k": 5, "pad": 2},
        {"type": "relu"},
        {"type": "maxpool2d", "k": 2},
        {"name": "conv2", "type": "conv2d", "in": c1, "out": c2, "k": 5, "pad": 2},
        {"type": "relu"},
        {"type": "maxpool2d", "k": 2},
        {"name": "conv3", "type": "conv2d", "in": c2, "out": c3, "k": 3, "pad": 1},
        {"type": "relu"},
    ]
    if arch == "cnn_fc":
        side = CANVAS // 4
        return [
            *trunk,
            {"type": "flatten"},
            linear("fc1", c3 * side * side, FC_HIDDEN),
            {"type": "relu"},
            linear("fc2", FC_HIDDEN, N_CLASSES),
        ]
    if arch == "cnn_gap":
        return [
            *trunk,
            {"type": "gap"},
            {"type": "flatten"},
            linear("fc1", c3, N_CLASSES),
        ]
    raise ValueError(f"unknown arch {arch!r}")


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 4096):
    """Predicted labels and max-softmax confidence for canvases x (N, 56, 56)."""
    preds, confs = [], []
    for i in range(0, len(x), batch_size):
        probs = model(x[i : i + batch_size]).softmax(dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(pred)
        confs.append(conf)
    return torch.cat(preds), torch.cat(confs)


def round_to_fp16_(model: nn.Module) -> nn.Module:
    """Round all parameters to float16 and back to float32, in place (exported weights)."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())
    return model


def load_checkpoint(path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = build_model(cfg["arch"], tuple(cfg["channels"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg
