#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pathlib

from PIL import Image
from tensorboard.backend.event_processing import event_accumulator


def _load_image_events(logdir: pathlib.Path) -> dict[str, list]:
    acc = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.IMAGES: 0},
    )
    acc.Reload()
    return {tag: list(acc.Images(tag)) for tag in acc.Tags().get("images", [])}


def _pick_tag(images_by_tag: dict[str, list], tag_suffix: str, episode: str) -> str:
    suffix = f"/{tag_suffix}"
    episode_part = f"/episode_{episode}/"
    candidates = [
        tag
        for tag in images_by_tag
        if tag.endswith(suffix) and episode_part in f"/{tag}"
    ]
    if not candidates:
        candidates = [tag for tag in images_by_tag if tag.endswith(suffix)]
    if not candidates:
        available = ", ".join(sorted(images_by_tag))
        raise SystemExit(f"No image tag ending in /{tag_suffix}. Available: {available}")
    return sorted(candidates)[0]


def _event_to_rgb(event) -> Image.Image:
    return Image.open(io.BytesIO(event.encoded_image_string)).convert("RGB")


def _fit_to_canvas(img: Image.Image, width: int, height: int) -> Image.Image:
    scaled = img.resize((width, height), Image.Resampling.NEAREST)
    return scaled


def _with_alpha(img: Image.Image, opacity: float, alpha_floor: int) -> Image.Image:
    rgba = img.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    opacity = max(0.0, min(1.0, opacity))
    for y in range(height):
        for x in range(width):
            r, g, b, _a = pixels[x, y]
            if r <= 2 and g <= 2 and b >= 250:
                pixels[x, y] = (0, 0, 0, 0)
                continue
            brightness = max(r, g, b)
            alpha = int(max(alpha_floor, brightness) * opacity)
            if brightness <= 0:
                alpha = 0
            pixels[x, y] = (r, g, b, max(0, min(255, alpha)))
    return rgba


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="threat")
    parser.add_argument("--episode", default="000")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--opacity", type=float, default=0.70)
    parser.add_argument(
        "--alpha-floor",
        type=int,
        default=32,
        help="Minimum alpha for nonblack heatmap pixels before opacity is applied.",
    )
    parser.add_argument("--metadata-out")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_tag = _load_image_events(logdir)
    full_tag = _pick_tag(images_by_tag, args.tag, args.episode)
    events = sorted(images_by_tag[full_tag], key=lambda event: int(event.step))
    if not events:
        raise SystemExit(f"No image events for tag {full_tag}")

    steps = []
    for idx, event in enumerate(events):
        step = int(event.step)
        steps.append(step)
        rgb = _fit_to_canvas(_event_to_rgb(event), args.width, args.height)
        rgb.save(out_dir / f"{args.tag}_rgb_{idx:06d}.png")
        rgba = _with_alpha(rgb, args.opacity, args.alpha_floor)
        rgba.save(out_dir / f"{args.tag}_overlay_{idx:06d}.png")

    metadata = {
        "logdir": str(logdir),
        "tag": full_tag,
        "frame_count": len(events),
        "steps": steps,
        "width": args.width,
        "height": args.height,
        "opacity": args.opacity,
        "alpha_floor": args.alpha_floor,
        "rgb_pattern": str(out_dir / f"{args.tag}_rgb_%06d.png"),
        "overlay_pattern": str(out_dir / f"{args.tag}_overlay_%06d.png"),
    }
    if args.metadata_out:
        pathlib.Path(args.metadata_out).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    else:
        print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
