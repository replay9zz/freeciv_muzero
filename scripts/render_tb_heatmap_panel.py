#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pathlib
from collections import defaultdict

from PIL import Image, ImageDraw
from tensorboard.backend.event_processing import event_accumulator


def _parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _load_images(logdir: pathlib.Path) -> dict[str, list]:
    acc = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.IMAGES: 0},
    )
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("images", []):
        out[tag] = list(acc.Images(tag))
    return out


def _pick_tag(images_by_tag: dict[str, list], short_tag: str, episode: str) -> str | None:
    suffix = f"/{short_tag}"
    episode_part = f"/episode_{episode}/"
    candidates = [
        tag
        for tag in images_by_tag
        if tag.endswith(suffix) and episode_part in f"/{tag}"
    ]
    if not candidates:
        candidates = [tag for tag in images_by_tag if tag.endswith(suffix)]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _latest_by_step(events: list) -> dict[int, object]:
    latest = {}
    for event in events:
        latest[int(event.step)] = event
    return latest


def _event_to_image(event) -> Image.Image:
    return _mute_empty_heatmap_blue(
        Image.open(io.BytesIO(event.encoded_image_string)).convert("RGB")
    )


def _mute_empty_heatmap_blue(img: Image.Image) -> Image.Image:
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r <= 2 and g <= 2 and b >= 250:
                pixels[x, y] = (0, 0, 0)
    return img


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    pad = 5
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(245, 245, 245))


def _fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    return img.resize((max(1, max_w), max(1, max_h)), Image.Resampling.NEAREST)


def _render_panel(
    images_for_step: dict[str, Image.Image | None],
    step: int,
    width: int,
    height: int,
    cols: int,
) -> Image.Image:
    rows = max(1, (len(images_for_step) + cols - 1) // cols)
    title_h = 34
    cell_w = width // cols
    cell_h = max(1, (height - title_h) // rows)
    panel = Image.new("RGB", (width, height), (16, 18, 20))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 9), f"Belief heatmaps  turn={step}", fill=(245, 245, 245))

    for idx, (name, img) in enumerate(images_for_step.items()):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = title_h + row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(60, 64, 70))
        _draw_label(draw, (x0 + 10, y0 + 10), name)
        if img is None:
            draw.text((x0 + 10, y0 + 40), "no frame", fill=(180, 180, 180))
            continue
        fitted = _fit_image(img, cell_w - 24, cell_h - 50)
        px = x0 + (cell_w - fitted.width) // 2
        py = y0 + 42 + max(0, (cell_h - 52 - fitted.height) // 2)
        panel.paste(fitted, (px, py))

    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tags",
        default="belief_units,threat,visible_units,territory",
        help="Comma-separated heatmap tag suffixes.",
    )
    parser.add_argument("--episode", default="000")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--metadata-out")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    out_dir = pathlib.Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_tag = _load_images(logdir)
    requested_tags = _parse_tags(args.tags)
    selected = {
        short: _pick_tag(images_by_tag, short, args.episode)
        for short in requested_tags
    }
    selected = {short: tag for short, tag in selected.items() if tag is not None}
    if not selected:
        raise SystemExit(f"No matching TensorBoard image tags found under {logdir}")

    by_short_tag = {
        short: _latest_by_step(images_by_tag[full_tag])
        for short, full_tag in selected.items()
    }
    steps = sorted({step for events in by_short_tag.values() for step in events})
    if not steps:
        raise SystemExit(f"No TensorBoard image frames found under {logdir}")

    last_seen: dict[str, Image.Image | None] = defaultdict(lambda: None)
    for frame_idx, step in enumerate(steps):
        current = {}
        for short in requested_tags:
            event = by_short_tag.get(short, {}).get(step)
            if event is not None:
                last_seen[short] = _event_to_image(event)
            current[short] = last_seen[short]
        panel = _render_panel(current, step, args.width, args.height, args.cols)
        panel.save(out_dir / f"frame_{frame_idx:06d}.png")

    metadata = {
        "frame_count": len(steps),
        "steps": steps,
        "selected_tags": selected,
        "requested_tags": requested_tags,
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
