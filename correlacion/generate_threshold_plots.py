#!/usr/bin/env python3
"""Generate combined threshold plot images from correlacion measure PNGs.

Usage examples:
    python correlacion/generate_threshold_plots.py \
        --parameter max_threshold \
        --item El_espejo \
        --text "First image text" \
        --text2 "Second image text"

This script builds two 2x2 collages:
    1) FFD, SFD, TFD + text block
    2) FPRT, SPRT, RRT + text block

It reads files from:
    correlacion/experiments/<item>/<parameter>/measures_correlation/

and writes outputs to the same folder by default.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEASURES_GROUPS = [
    ("group1", ["FFD", "SFD", "TFD"]),
    ("group2", ["FPRT", "SPRT", "RRT"]),
]
DEFAULT_CELL_SIZE = (2100, 1500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine measure plots into two images with text blocks.")
    parser.add_argument("--parameter", required=True, help="Folder name under correlacion/experiments/<item> to read.")
    parser.add_argument("--item", default="El_espejo", help="Item folder name under correlacion/experiments. Defaults to El_espejo.")
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parent / "experiments"),
                        help="Root experiments directory. Defaults to correlacion/experiments.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Defaults to the measures_correlation folder inside the item/parameter directory.")
    parser.add_argument("--text", default="",
                        help="Text to place into the fourth cell for both output images.")
    parser.add_argument("--text1", default=None,
                        help="Text for the first output image. Overrides --text for image 1.")
    parser.add_argument("--text2", default=None,
                        help="Text for the second output image. Overrides --text for image 2.")
    parser.add_argument("--cell-width", type=int, default=DEFAULT_CELL_SIZE[0], help="Width of each cell in pixels.")
    parser.add_argument("--cell-height", type=int, default=DEFAULT_CELL_SIZE[1], help="Height of each cell in pixels.")
    parser.add_argument("--font-size", type=int, default=80, help="Base font size for text blocks.")
    parser.add_argument("--verbose", action="store_true", help="Print extra progress information.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_plot(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    if image.size == target_size:
        return image
    image = image.convert("RGB")
    image_w, image_h = image.size
    ratio = min(target_w / image_w, target_h / image_h)
    new_size = (int(image_w * ratio), int(image_h * ratio))
    resized = image.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, "white")
    x = (target_w - new_size[0]) // 2
    y = (target_h - new_size[1]) // 2
    canvas.paste(resized, (x, y))
    return canvas


def create_text_image(text: str, size: tuple[int, int], font: ImageFont.ImageFont) -> Image.Image:
    """Render text into an image, supporting literal '\\n' newlines and wrapping to fit.

    Literal backslash-n sequences ("\\n") passed via CLI will be converted to actual
    newlines. Text is wrapped by pixel width using the provided font.
    """
    width, height = size
    background = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(background)

    # convert literal backslash-n to actual newlines and ensure non-empty
    text = (text or "").replace("\\n", "\n").strip()
    if not text:
        text = "No text provided."

    margin = int(width * 0.05)
    max_text_width = width - margin * 2

    # compute line height using font metrics for compatibility
    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + 14
    except Exception:
        bbox = font.getbbox("A")
        line_height = (bbox[3] - bbox[1]) + 14

    # wrap text by pixel width while preserving explicit newlines
    lines: list[str] = []
    paras = text.split("\n")
    for pi, para in enumerate(paras):
        words = para.split()
        if not words:
            # preserve blank lines
            lines.append("")
        else:
            cur = words[0]
            for w in words[1:]:
                test = cur + " " + w
                try:
                    bbox = font.getbbox(test)
                    test_width = bbox[2] - bbox[0]
                except Exception:
                    # fallback: approximate width by characters
                    test_width = len(test) * (font.size // 2)

                if test_width <= max_text_width:
                    cur = test
                else:
                    lines.append(cur)
                    cur = w
            lines.append(cur)

        # Add an extra blank line between paragraphs for increased spacing
        if pi < len(paras) - 1:
            lines.append("")

    # vertical centering
    total_text_height = line_height * len(lines)
    y = max((height - total_text_height) // 2, margin)

    for line in lines:
        x = margin
        draw.text((x, y), line, fill="black", font=font)
        y += line_height

    return background


def find_plot_path(base_dir: Path, item: str, measure: str) -> Path:
    file_name = f"{item}_{measure}.png"
    candidate = base_dir / file_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing plot file: {candidate}")


def build_collage(images: list[Image.Image], text_image: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    cols, rows = 2, 2
    total_width = cell_size[0] * cols
    total_height = cell_size[1] * rows
    collage = Image.new("RGB", (total_width, total_height), "white")

    positions = [
        (0, 0),
        (cell_size[0], 0),
        (0, cell_size[1]),
        (cell_size[0], cell_size[1]),
    ]

    for idx, img in enumerate(images):
        collage.paste(img, positions[idx])

    collage.paste(text_image, positions[3])
    return collage


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    threshold_dir = root_dir / args.item / args.parameter
    measures_dir = threshold_dir / "measures_correlation"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else measures_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    text1 = args.text1 if args.text1 is not None else args.text
    text2 = args.text2 if args.text2 is not None else args.text
    font = load_font(args.font_size)
    cell_size = (args.cell_width, args.cell_height)

    if args.verbose:
        print(f"Root directory: {root_dir}")
        print(f"Threshold directory: {threshold_dir}")
        print(f"Measures directory: {measures_dir}")
        print(f"Output directory: {output_dir}")

    if not measures_dir.exists():
        raise FileNotFoundError(f"Measures directory not found: {measures_dir}")

    for group_name, measures in MEASURES_GROUPS:
        if args.verbose:
            print(f"Building '{group_name}' with measures: {measures}")

        images = []
        for measure in measures:
            plot_path = find_plot_path(measures_dir, args.item, measure)
            image = Image.open(plot_path)
            images.append(resize_plot(image, cell_size))
            if args.verbose:
                print(f"Loaded {plot_path.name} -> resized to {cell_size}")

        text_image = create_text_image(text1 if group_name == "group1" else text2, cell_size, font)
        collage = build_collage(images, text_image, cell_size)

        output_name = f"{args.item}_{group_name}.png"
        output_path = output_dir / output_name
        collage.save(output_path, dpi=(300, 300))

        # also save a copy under correlacion/plots/<item>/ as <parameter>_<item>_group_<n>.png
        try:
            group_idx = 1 if group_name == "group1" else 2
            plots_root = Path(__file__).resolve().parent / "plots" / args.item
            plots_root.mkdir(parents=True, exist_ok=True)
            plots_name = f"{args.parameter}_{args.item}_group_{group_idx}.png"
            plots_path = plots_root / plots_name
            collage.save(plots_path, dpi=(300, 300))
            if args.verbose:
                print(f"Saved duplicate collage to: {plots_path}")
        except Exception as e:
            if args.verbose:
                print(f"Warning: could not save duplicate collage: {e}")

        if args.verbose:
            print(f"Saved combined collage: {output_path}")

    print("Done. Generated two images:")
    print(f"  {output_dir / f'{args.item}_group1.png'}")
    print(f"  {output_dir / f'{args.item}_group2.png'}")


if __name__ == "__main__":
    main()
