#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    out = Path(__file__).resolve().parent / "sample_invoice.png"
    image = Image.new("RGB", (960, 540), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, 960, 84), fill=(26, 43, 64))
    draw.text((48, 24), "NEMOTRON OCR V2", fill="white", font=font(38))
    draw.text((48, 140), "vLLM plugin integration benchmark", fill=(20, 20, 20), font=font(30))
    draw.text((48, 210), "Invoice total: $42.19", fill=(20, 20, 20), font=font(36))
    draw.text((48, 290), "Status: verified direct vs vLLM output", fill=(20, 20, 20), font=font(28))
    draw.text((48, 366), "Batch size: 4 sample pages", fill=(20, 20, 20), font=font(28))

    draw.rectangle((690, 132, 894, 404), outline=(26, 43, 64), width=4)
    draw.text((720, 180), "OCR", fill=(26, 43, 64), font=font(46))
    draw.text((720, 250), "vLLM", fill=(26, 43, 64), font=font(38))

    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
