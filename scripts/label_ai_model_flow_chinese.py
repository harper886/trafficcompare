from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "model-flow-ai-background.png"
OUT = ROOT / "assets" / "model-flow-ai-chinese-labeled.png"

WHITE = (242, 247, 255)
MUTED = (173, 201, 232)
CYAN = (78, 224, 232)
BLUE = (95, 156, 255)
PURPLE = (164, 104, 255)
YELLOW = (255, 202, 82)
ORANGE = (255, 136, 58)
RED = (255, 84, 72)
DARK = (4, 12, 28)


def font(size, bold=False):
    path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    return ImageFont.truetype(path, size)


F_TITLE = font(58, True)
F_SUB = font(25)
F_HEAD = font(28, True)
F_BODY = font(18)


def draw_center(draw, box, text, fnt, fill=WHITE):
    x1, y1, x2, y2 = box
    tb = draw.textbbox((0, 0), text, font=fnt)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, font=fnt, fill=fill)


def label_box(draw, x, y, w, h, title, subtitle, color):
    # Dark glass label with colored top rule.
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=(3, 12, 28, 214), outline=(*color, 235), width=2)
    draw.line((x + 18, y + 14, x + w - 18, y + 14), fill=color, width=4)
    draw_center(draw, (x + 8, y + 18, x + w - 8, y + 55), title, F_HEAD, WHITE)
    if subtitle:
        draw_center(draw, (x + 8, y + 58, x + w - 8, y + h - 8), subtitle, F_BODY, MUTED)


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    img = Image.open(SRC).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Slight top and bottom scrims for legibility.
    top = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    td = ImageDraw.Draw(top)
    td.rectangle((0, 0, w, 180), fill=(0, 8, 22, 120))
    td.rectangle((0, h - 115, w, h), fill=(0, 8, 22, 110))
    overlay = Image.alpha_composite(overlay, top.filter(ImageFilter.GaussianBlur(6)))
    draw = ImageDraw.Draw(overlay)

    draw_center(draw, (0, 36, w, 105), "交通碰撞风险预测模型流程", F_TITLE, WHITE)
    draw_center(draw, (0, 108, w, 150), "基于过去5个时间步、三类空间邻接关系，预测每个区域的风险概率", F_SUB, MUTED)

    # Label positions fit the AI background's left-to-right panel layout.
    labels = [
        (60, 855, 245, 90, "历史交通输入", "过去5个时间步", BLUE),
        (360, 855, 250, 90, "区域特征表示", "116维交通特征", PURPLE),
        (680, 855, 255, 90, "动态演化", "平滑门减少突变", PURPLE),
        (975, 855, 290, 90, "三路空间注意力", "道路 / POI / 历史事故", CYAN),
        (1308, 855, 260, 90, "ConvLSTM建模", "融合时间与空间变化", YELLOW),
        (1605, 855, 255, 90, "风险概率与预警", "高风险区域输出", RED),
    ]
    for item in labels:
        label_box(draw, *item)

    # Small project-specific footer.
    footer = "核心思路：时间依赖由 ConvLSTM 学习，空间关联由多路注意力学习，输出端结合平滑策略减少预警抖动"
    draw_center(draw, (160, 990, w - 160, 1045), footer, font(23), (197, 221, 246))

    # Add subtle arrow captions above the flow if the AI diagram has empty mid space.
    mini = [
        (210, 168, "时间窗口"),
        (520, 168, "特征编码"),
        (850, 168, "状态更新"),
        (1165, 168, "空间传播"),
        (1480, 168, "概率输出"),
    ]
    for x, y, text in mini:
        draw.rounded_rectangle((x - 70, y, x + 70, y + 36), 18, fill=(5, 18, 42, 170), outline=(70, 130, 210, 160), width=1)
        draw_center(draw, (x - 70, y, x + 70, y + 36), text, font(17, True), (205, 224, 248))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
