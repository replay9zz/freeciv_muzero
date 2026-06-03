#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
from collections import defaultdict

from PIL import Image, ImageDraw
from tensorboard.backend.event_processing import event_accumulator

EMPTY_HEATMAP_COLOR = (224, 224, 224)
EMPTY_HEATMAP_OUTLINE = (174, 174, 174)


def _parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "heatmap"


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
    return _mute_empty_heatmap_default(
        Image.open(io.BytesIO(event.encoded_image_string)).convert("RGB")
    )


def _mute_empty_heatmap_default(img: Image.Image) -> Image.Image:
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r <= 2 and g <= 2 and b >= 250:
                pixels[x, y] = EMPTY_HEATMAP_COLOR
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


def _sample_tile_colors(
    img: Image.Image,
    map_w: int | None = None,
    map_h: int | None = None,
) -> tuple[list[list[tuple[int, int, int]]], int, int]:
    map_w = int(map_w or img.width)
    map_h = int(map_h or img.height)
    if map_w <= 0 or map_h <= 0:
        raise ValueError("map dimensions must be positive")

    pixels = img.load()
    colors: list[list[tuple[int, int, int]]] = []
    for y in range(map_h):
        row = []
        y0 = int(y * img.height / map_h)
        y1 = max(y0 + 1, int((y + 1) * img.height / map_h))
        for x in range(map_w):
            x0 = int(x * img.width / map_w)
            x1 = max(x0 + 1, int((x + 1) * img.width / map_w))
            total = [0, 0, 0]
            count = 0
            for py in range(y0, min(y1, img.height)):
                for px in range(x0, min(x1, img.width)):
                    r, g, b = pixels[px, py]
                    total[0] += r
                    total[1] += g
                    total[2] += b
                    count += 1
            if count:
                row.append(tuple(int(v / count) for v in total))
            else:
                row.append((0, 0, 0))
        colors.append(row)
    return colors, map_w, map_h


def _hex_points(cx: float, cy: float, tile_w: float, tile_h: float) -> list[tuple[int, int]]:
    side = tile_w * 0.23
    half_w = tile_w / 2.0
    half_h = tile_h / 2.0
    return [
        (round(cx - half_w + side), round(cy - half_h)),
        (round(cx + half_w - side), round(cy - half_h)),
        (round(cx + half_w), round(cy)),
        (round(cx + half_w - side), round(cy + half_h)),
        (round(cx - half_w + side), round(cy + half_h)),
        (round(cx - half_w), round(cy)),
    ]


def _render_hex_heatmap(
    img: Image.Image,
    max_w: int,
    max_h: int,
    map_w: int | None = None,
    map_h: int | None = None,
) -> Image.Image:
    colors, cols, rows = _sample_tile_colors(img, map_w, map_h)
    bg = (9, 12, 14)
    canvas = Image.new("RGB", (max(1, max_w), max(1, max_h)), bg)
    draw = ImageDraw.Draw(canvas)
    pad = 8
    avail_w = max(1, max_w - pad * 2)
    avail_h = max(1, max_h - pad * 2)
    row_step = 0.75
    width_units = cols + 0.5
    height_units = 1.0 + max(0, rows - 1) * row_step
    tile_w = min(avail_w / width_units, (avail_h / height_units) / 0.72)
    tile_h = tile_w * 0.72
    if tile_h * height_units > avail_h:
        tile_h = avail_h / height_units
        tile_w = tile_h / 0.72
    total_w = tile_w * width_units
    total_h = tile_h * height_units
    x_origin = (max_w - total_w) / 2.0
    y_origin = (max_h - total_h) / 2.0

    # Draw dark base first so adjacent native rows read as a Freeciv-style hex map.
    for y in range(rows):
        for x in range(cols):
            cx = x_origin + (x + 0.5 + (0.5 if y % 2 else 0.0)) * tile_w
            cy = y_origin + (0.5 + y * row_step) * tile_h
            fill = colors[y][x]
            outline = tuple(max(0, int(c * 0.55)) for c in fill)
            if max(fill) <= 2:
                fill = EMPTY_HEATMAP_COLOR
                outline = EMPTY_HEATMAP_OUTLINE
            draw.polygon(
                _hex_points(cx, cy, tile_w * 0.96, tile_h * 0.90),
                fill=fill,
                outline=outline,
            )

    # Subtle native coordinate grid: y +/- 2 vertical, y +/- 1 diagonals.
    border = (82, 88, 92)
    draw.rectangle(
        (
            round(x_origin),
            round(y_origin),
            round(x_origin + total_w),
            round(y_origin + total_h),
        ),
        outline=border,
    )
    return canvas


def _render_panel(
    images_for_step: dict[str, Image.Image | None],
    step: int,
    width: int,
    height: int,
    cols: int,
    tile_shape: str,
    map_w: int | None,
    map_h: int | None,
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
        if tile_shape == "hex":
            fitted = _render_hex_heatmap(img, cell_w - 24, cell_h - 50, map_w, map_h)
        else:
            fitted = _fit_image(img, cell_w - 24, cell_h - 50)
        px = x0 + (cell_w - fitted.width) // 2
        py = y0 + 42 + max(0, (cell_h - 52 - fitted.height) // 2)
        panel.paste(fitted, (px, py))

    return panel


def _render_single(
    name: str,
    img: Image.Image | None,
    step: int,
    width: int,
    height: int,
    tile_shape: str,
    map_w: int | None,
    map_h: int | None,
) -> Image.Image:
    title_h = 34
    panel = Image.new("RGB", (width, height), (16, 18, 20))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 9), f"{name}  turn={step}", fill=(245, 245, 245))
    draw.rectangle((0, title_h, width - 1, height - 1), outline=(60, 64, 70))
    if img is None:
        draw.text((12, title_h + 12), "no frame", fill=(180, 180, 180))
        return panel
    max_w = max(1, width - 24)
    max_h = max(1, height - title_h - 24)
    if tile_shape == "hex":
        fitted = _render_hex_heatmap(img, max_w, max_h, map_w, map_h)
    else:
        fitted = _fit_image(img, max_w, max_h)
    px = (width - fitted.width) // 2
    py = title_h + (height - title_h - fitted.height) // 2
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
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Write one frame sequence per requested tag under out-dir/<tag>/.",
    )
    parser.add_argument("--tile-shape", choices=("square", "hex"), default="square")
    parser.add_argument("--map-width", type=int)
    parser.add_argument("--map-height", type=int)
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
        if args.separate:
            for short in requested_tags:
                tag_dir = out_dir / _safe_name(short)
                tag_dir.mkdir(parents=True, exist_ok=True)
                panel = _render_single(
                    short,
                    current[short],
                    step,
                    args.width,
                    args.height,
                    args.tile_shape,
                    args.map_width,
                    args.map_height,
                )
                panel.save(tag_dir / f"frame_{frame_idx:06d}.png")
        else:
            panel = _render_panel(
                current,
                step,
                args.width,
                args.height,
                args.cols,
                args.tile_shape,
                args.map_width,
                args.map_height,
            )
            panel.save(out_dir / f"frame_{frame_idx:06d}.png")

    metadata = {
        "frame_count": len(steps),
        "steps": steps,
        "selected_tags": selected,
        "requested_tags": requested_tags,
        "tile_shape": args.tile_shape,
        "map_width": args.map_width,
        "map_height": args.map_height,
        "mode": "separate" if args.separate else "panel",
        "frame_dirs": {
            short: str(out_dir / _safe_name(short)) for short in requested_tags
        }
        if args.separate
        else {},
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
