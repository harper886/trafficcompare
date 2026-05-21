from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "chinese-model-flow-traffic-risk.png"

W, H = 1920, 1080
BG = (4, 12, 28)
WHITE = (238, 244, 252)
MUTED = (154, 184, 218)
BLUE = (70, 145, 255)
CYAN = (55, 224, 224)
PURPLE = (153, 89, 246)
YELLOW = (255, 196, 62)
ORANGE = (255, 123, 52)
RED = (255, 78, 68)


def font(size, bold=False):
    path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    return ImageFont.truetype(path, size)


F_TITLE = font(52, True)
F_HEAD = font(28, True)
F_BODY = font(20)
F_SMALL = font(17)
F_NUM = font(42, True)


def rounded_rect(draw, xy, radius, fill, outline, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def glow_rect(base, xy, radius, color, blur=16, width=4):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, alpha in [(0, 100), (5, 60), (11, 34)]:
        d.rounded_rectangle(
            (xy[0] - i, xy[1] - i, xy[2] + i, xy[3] + i),
            radius=radius + i,
            outline=(*color, alpha),
            width=width,
        )
    return Image.alpha_composite(base, layer.filter(ImageFilter.GaussianBlur(blur)))


def text_center(draw, xy, text, fnt, fill=WHITE):
    x1, y1, x2, y2 = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, font=fnt, fill=fill)


def arrow(draw, x1, y1, x2, y2, color):
    draw.line((x1, y1, x2, y2), fill=color, width=5)
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 18
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - 0.45), y2 - head * math.sin(ang - 0.45)),
        (x2 - head * math.cos(ang + 0.45), y2 - head * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def panel(draw, img, x, y, w, h, title, color):
    img = glow_rect(img, (x, y, x + w, y + h), 22, color, blur=14)
    draw = ImageDraw.Draw(img)
    rounded_rect(draw, (x, y, x + w, y + h), 22, (5, 18, 40), color, 2)
    draw.rounded_rectangle((x, y, x + w, y + 64), radius=22, fill=(*color[:3],) if False else (18, 48, 88), outline=color, width=2)
    draw.rectangle((x, y + 38, x + w, y + 64), fill=(18, 48, 88))
    draw.line((x, y + 64, x + w, y + 64), fill=color, width=2)
    text_center(draw, (x, y + 4, x + w, y + 62), title, F_HEAD)
    return img, draw


def draw_grid(draw, x, y, cell=18, rows=8, cols=8, hot=None):
    hot = hot or {}
    for r in range(rows):
        for c in range(cols):
            fill = (18, 48, 88)
            if (r, c) in hot:
                fill = hot[(r, c)]
            draw.rectangle((x + c * cell, y + r * cell, x + (c + 1) * cell - 2, y + (r + 1) * cell - 2), fill=fill, outline=(56, 98, 150))


def draw_signal(draw, x, y, w=110, h=56, color=(80, 200, 255)):
    pts = []
    for i in range(7):
        px = x + i * w / 6
        py = y + h * (0.65 - 0.22 * math.sin(i * 1.35))
        pts.append((px, py))
    draw.rounded_rectangle((x - 8, y - 5, x + w + 8, y + h + 5), radius=8, outline=(56, 108, 166), width=1)
    draw.line(pts, fill=color, width=3)
    for p in pts:
        draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=color)


def main():
    img = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)

    # Background gradient and subtle circuit lines.
    vignette = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vignette)
    for i in range(28):
        a = max(0, 120 - i * 3)
        vd.rounded_rectangle((220 + i * 20, 80 + i * 10, W - 220 - i * 20, H - 80 - i * 8), 60, fill=a)
    tint = Image.new("RGBA", (W, H), (8, 28, 58, 255))
    img = Image.composite(tint, img, vignette.filter(ImageFilter.GaussianBlur(120)))
    draw = ImageDraw.Draw(img)

    title = "交通碰撞风险预测模型流程"
    text_center(draw, (0, 42, W, 112), title, F_TITLE, WHITE)
    draw.text((664, 120), "过去5个时间步 + 三类空间邻接图 → 输出每个区域的风险概率", font=F_BODY, fill=MUTED)
    draw.line((300, 88, 505, 88), fill=(48, 102, 190), width=2)
    draw.line((1415, 88, 1620, 88), fill=(48, 102, 190), width=2)

    panels = [
        (42, 210, 260, 650, "输入数据", BLUE),
        (350, 210, 260, 650, "区域特征表示", PURPLE),
        (658, 210, 300, 650, "动态演化 + 空间注意力", PURPLE),
        (1006, 210, 270, 650, "ConvLSTM 时空建模", CYAN),
        (1324, 210, 230, 650, "风险概率", BLUE),
        (1600, 210, 260, 650, "预警输出", RED),
    ]

    panel_boxes = []
    for x, y, w, h, name, color in panels:
        img, draw = panel(draw, img, x, y, w, h, name, color)
        panel_boxes.append((x, y, w, h, color))

    # Panel 1: inputs.
    x, y, w, h, _ = panel_boxes[0]
    for i, label in enumerate(["t-4", "t-3", "...", "t"]):
        yy = y + 105 + i * 112
        draw.text((x + 18, yy + 22), label, font=F_BODY, fill=WHITE)
        if label != "...":
            draw_signal(draw, x + 70, yy, 92, 48)
            draw_grid(draw, x + 176, yy - 5, cell=13, rows=5, cols=5, hot={(2, 3): ORANGE, (3, 2): YELLOW})
        else:
            draw.text((x + 110, yy + 18), "......", font=F_NUM, fill=MUTED)
    draw.text((x + 38, y + h - 84), "交通时序特征", font=F_SMALL, fill=MUTED)
    draw.text((x + 38, y + h - 50), "区域网格标签", font=F_SMALL, fill=MUTED)

    # Panel 2: feature representation.
    x, y, w, h, _ = panel_boxes[1]
    for i in range(4):
        yy = y + 120 + i * 120
        draw.rounded_rectangle((x + 34, yy, x + 62, yy + 64), 4, fill=(38, 90, 178), outline=(86, 148, 255))
        arrow(draw, x + 76, yy + 32, x + 116, yy + 32, (80, 145, 255))
        draw.rounded_rectangle((x + 125, yy + 12, x + 224, yy + 52), 10, fill=(29, 21, 70), outline=PURPLE, width=2)
        for k in range(5):
            draw.ellipse((x + 140 + k * 16, yy + 25, x + 151 + k * 16, yy + 36), fill=(143, 98, 245))
    draw.text((x + 38, y + h - 84), "116维交通特征", font=F_SMALL, fill=MUTED)
    draw.text((x + 38, y + h - 50), "转成模型可学习表示", font=F_SMALL, fill=MUTED)

    # Panel 3: evolution and multi attention.
    x, y, w, h, _ = panel_boxes[2]
    draw.rounded_rectangle((x + 24, y + 96, x + w - 24, y + 220), 14, fill=(18, 26, 60), outline=PURPLE, width=2)
    draw.text((x + 44, y + 114), "Evolution 动态演化", font=F_BODY, fill=WHITE)
    draw.text((x + 44, y + 150), "逐时间步更新风险状态", font=F_SMALL, fill=MUTED)
    draw.text((x + 44, y + 180), "平滑门减少特征突变", font=F_SMALL, fill=MUTED)
    for i, name in enumerate(["道路邻接", "POI邻接", "历史事故"]):
        yy = y + 280 + i * 92
        draw.rounded_rectangle((x + 30, yy, x + 126, yy + 46), 10, fill=(28, 22, 72), outline=PURPLE, width=2)
        text_center(draw, (x + 30, yy, x + 126, yy + 46), name, F_SMALL)
        for j in range(5):
            px = x + 175 + (j % 2) * 56
            py = yy - 6 + j * 12
            draw.ellipse((px, py, px + 12, py + 12), fill=(142, 88, 236))
        for a in range(4):
            draw.line((x + 130, yy + 23, x + 178 + (a % 2) * 56, yy + 3 + a * 14), fill=(142, 88, 236), width=1)
    draw.text((x + 70, y + h - 64), "三路注意力独立学习空间影响", font=F_SMALL, fill=MUTED)

    # Panel 4: ConvLSTM.
    x, y, w, h, _ = panel_boxes[3]
    for i in range(5):
        yy = y + 110 + i * 86
        draw.rounded_rectangle((x + 78, yy, x + 162, yy + 46), 9, fill=(10, 58, 70), outline=CYAN, width=2)
        text_center(draw, (x + 78, yy, x + 162, yy + 46), "ConvLSTM", F_SMALL)
        if i < 4:
            arrow(draw, x + 120, yy + 52, x + 120, yy + 78, CYAN)
    draw.rounded_rectangle((x + 198, y + 130, x + 232, y + 566), 5, fill=(9, 55, 66), outline=CYAN, width=2)
    for i in range(8):
        c = 60 + i * 14
        draw.rectangle((x + 202, y + 136 + i * 48, x + 228, y + 171 + i * 48), fill=(25, c, c + 30))
    draw.text((x + 46, y + h - 92), "融合动态特征与静态特征", font=F_SMALL, fill=MUTED)
    draw.text((x + 46, y + h - 58), "学习时空变化规律", font=F_SMALL, fill=MUTED)

    # Panel 5: probability.
    x, y, w, h, _ = panel_boxes[4]
    bar_x = x + 88
    draw.rounded_rectangle((bar_x, y + 145, bar_x + 56, y + 500), 14, fill=(9, 18, 38), outline=(170, 200, 235), width=2)
    for i in range(9):
        t = i / 8
        col = (
            int(28 + t * 220),
            int(65 + t * 20),
            int(100 - t * 55),
        )
        draw.rectangle((bar_x + 6, y + 482 - i * 37, bar_x + 50, y + 514 - i * 37), fill=col)
    draw.text((x + 60, y + 560), "0.87", font=F_NUM, fill=RED)
    draw.text((x + 48, y + 626), "区域风险概率", font=F_BODY, fill=WHITE)
    draw.text((x + 42, y + h - 64), "数值越高，风险越大", font=F_SMALL, fill=MUTED)

    # Panel 6: warning.
    x, y, w, h, _ = panel_boxes[5]
    warn = [(x + 130, y + 160), (x + 62, y + 292), (x + 198, y + 292)]
    draw.line([warn[0], warn[1], warn[2], warn[0]], fill=RED, width=10, joint="curve")
    draw.line([warn[0], warn[1], warn[2], warn[0]], fill=(255, 156, 132), width=3)
    draw.text((x + 118, y + 190), "!", font=font(74, True), fill=(255, 150, 130))
    draw.text((x + 52, y + 352), "高风险区域", font=font(33, True), fill=RED)
    draw.text((x + 52, y + 410), "触发预警", font=font(33, True), fill=RED)
    draw.rounded_rectangle((x + 48, y + 520, x + 212, y + 594), 12, fill=(48, 14, 20), outline=RED, width=2)
    text_center(draw, (x + 48, y + 520, x + 212, y + 594), "TP / FP / FN", F_BODY, (255, 188, 172))
    draw.text((x + 34, y + h - 64), "用于前端标签校验与解释", font=F_SMALL, fill=MUTED)

    # Arrows between panels.
    draw = ImageDraw.Draw(img)
    for i in range(len(panel_boxes) - 1):
        x, y, w, h, color = panel_boxes[i]
        nx, ny, nw, nh, ncolor = panel_boxes[i + 1]
        arrow(draw, x + w + 10, y + h / 2, nx - 14, ny + nh / 2, color)

    # Footer.
    draw.text((515, 914), "模型要点：LSTM/ConvLSTM 负责时间变化，Attention 负责空间关联，平滑与后处理减少预警抖动。", font=F_BODY, fill=(174, 200, 226))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
