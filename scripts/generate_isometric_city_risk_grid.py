from pathlib import Path
from math import sin

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "isometric-city-risk-grid.png"


W, H = 1920, 1080
BG = (5, 12, 26)
ROAD = (44, 64, 88)
ROAD_SOFT = (18, 31, 50)
BLOCK_TOP = (12, 34, 64)
BLOCK_LEFT = (8, 24, 46)
BLOCK_RIGHT = (6, 20, 40)
EDGE = (46, 83, 116)


def iso(cx, cy, gx, gy, tile_w=112, tile_h=58):
    return (
        cx + (gx - gy) * tile_w / 2,
        cy + (gx + gy) * tile_h / 2,
    )


def diamond(draw, x, y, w, h, fill, outline=None):
    pts = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]
    draw.polygon(pts, fill=fill, outline=outline)
    return pts


def block(draw, x, y, w, h, height, top, left, right, outline=EDGE):
    top_pts = [(x, y - h / 2 - height), (x + w / 2, y - height), (x, y + h / 2 - height), (x - w / 2, y - height)]
    left_pts = [(x - w / 2, y - height), (x, y + h / 2 - height), (x, y + h / 2), (x - w / 2, y)]
    right_pts = [(x + w / 2, y - height), (x, y + h / 2 - height), (x, y + h / 2), (x + w / 2, y)]
    draw.polygon(left_pts, fill=left)
    draw.polygon(right_pts, fill=right)
    draw.polygon(top_pts, fill=top, outline=outline)
    draw.line([top_pts[0], top_pts[1], top_pts[2], top_pts[3], top_pts[0]], fill=outline, width=1)


def glow_layer(points, color, blur):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x, y, w, h, strength in points:
        for scale, alpha in [(1.2, int(92 * strength)), (1.8, int(56 * strength)), (2.4, int(32 * strength))]:
            diamond(d, x, y, w * scale, h * scale, (*color, alpha))
    return layer.filter(ImageFilter.GaussianBlur(blur))


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Subtle background vignette.
    vignette = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vignette)
    for i in range(0, 900, 18):
        alpha = max(0, 90 - i // 10)
        vd.ellipse((W / 2 - i * 1.6, H / 2 - i, W / 2 + i * 1.6, H / 2 + i), fill=alpha)
    tint = Image.new("RGB", (W, H), (11, 27, 48))
    img = Image.composite(tint, img, vignette.filter(ImageFilter.GaussianBlur(90)))
    draw = ImageDraw.Draw(img)

    cx, cy = W / 2, 146
    tile_w, tile_h = 112, 58
    road_segments = []

    for i in range(9):
        x1, y1 = iso(cx, cy, i, 0, tile_w, tile_h)
        x2, y2 = iso(cx, cy, i, 8, tile_w, tile_h)
        road_segments.append((x1, y1 + 34, x2, y2 + 34))
        x3, y3 = iso(cx, cy, 0, i, tile_w, tile_h)
        x4, y4 = iso(cx, cy, 8, i, tile_w, tile_h)
        road_segments.append((x3, y3 + 34, x4, y4 + 34))

    for x1, y1, x2, y2 in road_segments:
        draw.line((x1, y1, x2, y2), fill=ROAD_SOFT, width=10)
    for x1, y1, x2, y2 in road_segments:
        draw.line((x1, y1, x2, y2), fill=ROAD, width=2)

    risk = {
        (1, 5): ((255, 103, 39), 1.0),
        (2, 2): ((239, 67, 42), 0.95),
        (4, 6): ((255, 152, 54), 0.85),
        (5, 1): ((221, 55, 48), 0.9),
        (6, 4): ((255, 129, 35), 0.78),
    }

    glow_points = []
    for gx, gy in risk:
        x, y = iso(cx, cy, gx + 0.5, gy + 0.5, tile_w, tile_h)
        glow_points.append((x, y + 12, tile_w * 0.78, tile_h * 0.78, risk[(gx, gy)][1]))

    img = Image.alpha_composite(img.convert("RGBA"), glow_layer(glow_points, (255, 84, 32), 22))
    draw = ImageDraw.Draw(img)

    for s in range(15, -1, -1):
        for gx in range(8):
            gy = s - gx
            if gy < 0 or gy > 7:
                continue
            x, y = iso(cx, cy, gx + 0.5, gy + 0.5, tile_w, tile_h)
            height = 20 + ((gx * 17 + gy * 11) % 34)
            wobble = int(5 * sin((gx + 1) * (gy + 2)))
            height += wobble
            if (gx, gy) in risk:
                color, strength = risk[(gx, gy)]
                top = tuple(min(255, int(c * 0.9 + 22)) for c in color)
                left = (118, 34, 30)
                right = (84, 28, 35)
                outline = (255, 178, 92)
                height += 14
            else:
                shade = (gx * 5 + gy * 7) % 18
                top = (BLOCK_TOP[0] + shade, BLOCK_TOP[1] + shade, BLOCK_TOP[2] + shade)
                left = (BLOCK_LEFT[0] + shade // 2, BLOCK_LEFT[1] + shade // 2, BLOCK_LEFT[2] + shade // 2)
                right = (BLOCK_RIGHT[0] + shade // 2, BLOCK_RIGHT[1] + shade // 2, BLOCK_RIGHT[2] + shade // 2)
                outline = EDGE
            block(draw, x, y + 28, tile_w * 0.72, tile_h * 0.72, height, top, left, right, outline)

    # Fine academic-grid accents.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2 in road_segments[::2]:
        od.line((x1, y1, x2, y2), fill=(100, 143, 176, 28), width=1)
    img = Image.alpha_composite(img, overlay)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
