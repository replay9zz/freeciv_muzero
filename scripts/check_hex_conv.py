#!/usr/bin/env python3
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeciv_sim.state.movement import FreecivMovement
from models import FreecivHexConv2d


def _expected_neighbors(height: int, width: int):
    movement = FreecivMovement(map_width=width, map_height=height)
    expected = {}
    for y in range(height):
        for x in range(width):
            expected[(x, y)] = [
                (x, y),
                *movement.get_native_neighbors(x, y),
            ]
    return expected


def main():
    height, width = 8, 5
    x = torch.arange(height * width, dtype=torch.float32).view(1, 1, height, width)
    layer = FreecivHexConv2d(1, 7, bias=False)
    with torch.no_grad():
        layer.proj.weight.zero_()
        for out_ch in range(7):
            layer.proj.weight[out_ch, out_ch, 0, 0] = 1.0

    y = layer(x)[0]
    expected = _expected_neighbors(height, width)

    for py in range(height):
        for px in range(width):
            want = []
            for nx, ny in expected[(px, py)]:
                if nx is None or ny is None:
                    want.append(0.0)
                else:
                    want.append(float(x[0, 0, ny, nx]))
            got = y[:, py, px].tolist()
            if got != want:
                raise AssertionError(
                    f"Mismatch at {(px, py)}: got={got} want={want}"
                )

    print("FreecivHexConv2d neighbor mapping OK")


if __name__ == "__main__":
    main()
