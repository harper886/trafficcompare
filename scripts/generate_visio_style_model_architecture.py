from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "visio-style-model-architecture.png"

W, H = 1700, 980


def font(size, bold=False):
    if bold:
        for p in [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/msyhbd.ttc",
        ]:
            if Path(p).exists():
                return ImageFont.truetype(p, size)
    else:
        for p in [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/msyh.ttc",
        ]:
            if Path(p).exists():
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = font(40, True)
F_SUB = font(18)
F_HEAD = font(21, True)
F_BODY = font(16)
F_SMALL = font(15)
F_FOOT = font(15)

BG = (255, 255, 255)
TEXT = (34, 42, 53)
SUB = (95, 106, 121)
GRID = (232, 236, 241)
BOX = (250, 252, 255)
BOX2 = (247, 250, 255)
LINE = (109, 128, 154)
BLUE = (68, 125, 214)
GREEN = (66, 160, 107)
ORANGE = (222, 148, 54)
RED = (217, 89, 93)
PURPLE = (131, 104, 214)


def text_center(draw, box, text, fnt, fill=TEXT):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 1), text, font=fnt, fill=fill)


def wrap_text(draw, text, fnt, max_width):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        test = w if not cur else cur + " " + w
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def rounded_box(img, box, fill, outline, width=2, radius=12, shadow=True):
    if shadow:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        x1, y1, x2, y2 = box
        d.rounded_rectangle((x1 + 4, y1 + 5, x2 + 4, y2 + 5), radius=radius, fill=(0, 0, 0, 22))
        img = Image.alpha_composite(img, layer.filter(ImageFilter.GaussianBlur(5)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    return img, d


def arrow(draw, start, end, color=LINE, width=3):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 > x1:
            pts = [(x2, y2), (x2 - 12, y2 - 6), (x2 - 12, y2 + 6)]
        else:
            pts = [(x2, y2), (x2 + 12, y2 - 6), (x2 + 12, y2 + 6)]
    else:
        if y2 > y1:
            pts = [(x2, y2), (x2 - 6, y2 - 12), (x2 + 6, y2 - 12)]
        else:
            pts = [(x2, y2), (x2 - 6, y2 + 12), (x2 + 6, y2 + 12)]
    draw.polygon(pts, fill=color)


def bullet_lines(draw, x, y, lines, fnt, color=TEXT, bullet=BLUE, line_h=22, max_width=0):
    cy = y
    for line in lines:
        wrap = wrap_text(draw, line, fnt, max_width) if max_width else [line]
        draw.ellipse((x, cy + 6, x + 7, cy + 13), fill=bullet)
        for i, seg in enumerate(wrap):
            draw.text((x + 16, cy + i * line_h), seg, font=fnt, fill=color)
        cy += line_h * len(wrap) + 4
    return cy


def tag(draw, x, y, text, color):
    draw.rounded_rectangle((x, y, x + 32, y + 26), radius=8, fill=color, outline=color)
    text_center(draw, (x, y, x + 32, y + 26), text, font(15, True), (255, 255, 255))


def main():
    img = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)

    # faint grid
    for gx in range(0, W, 40):
        draw.line((gx, 0, gx, H), fill=GRID, width=1)
    for gy in range(0, H, 40):
        draw.line((0, gy, W, gy), fill=GRID, width=1)

    draw.text((74, 36), "交通碰撞风险预测模型总体架构", font=F_TITLE, fill=TEXT)
    draw.text((76, 82), "图3-1  依据代码流程重绘的模型结构图", font=F_SUB, fill=SUB)

    # top flow boxes
    y = 170
    w = 250
    h = 205
    gap = 36
    xs = [70 + i * (w + gap) for i in range(5)]

    modules = [
        (
            "1",
            "输入构建",
            ["交通时序特征 X", "事故标签 y", "阈值信号 threshold_nc", "区域映射 dict_xy", "滑动窗口：5步历史"],
            BLUE,
            BOX,
        ),
        (
            "2",
            "动态特征演化",
            ["Evolution 层", "Update / Decay", "自适应平滑门", "输出动态状态 H_dynamic"],
            PURPLE,
            BOX2,
        ),
        (
            "3",
            "多源空间注意力",
            ["Road / POI / Record", "三类邻接关系", "Scaled Dot Attention", "邻居特征加权融合"],
            GREEN,
            BOX,
        ),
        (
            "4",
            "时空融合建模",
            ["动态特征 + 静态特征", "ConvLSTM2D", "时空特征融合", "学习区域风险表示"],
            ORANGE,
            BOX2,
        ),
        (
            "5",
            "风险预测输出",
            ["Dense + Sigmoid", "区域风险概率", "阈值判定", "Streaming 后处理"],
            RED,
            BOX,
        ),
    ]

    for i, (num, title, lines, color, fill) in enumerate(modules):
        x = xs[i]
        img, draw = rounded_box(img, (x, y, x + w, y + h), fill, color, width=2, radius=12, shadow=True)
        tag(draw, x + 14, y + 14, num, color)
        draw.text((x + 60, y + 16), title, font=F_HEAD, fill=TEXT)
        draw.line((x + 18, y + 54, x + w - 18, y + 54), fill=color, width=2)
        bullet_lines(draw, x + 28, y + 70, lines, F_BODY, max_width=w - 54)
        if i < 4:
            arrow(draw, (x + w + 10, y + h / 2), (xs[i + 1] - 12, y + h / 2), LINE, 3)

    # bottom detail boxes
    detail_y = 450
    bottom = [
        (70, 340, 480, "模型训练与评估", BLUE, [
            "训练目标：Focal Loss + 动态约束",
            "评价指标：AUC-PR、AUC-ROC、F1、Recall、Accuracy",
            "实验方式：基线对比 + 消融实验",
        ]),
        (575, 440, 560, "空间关系与注意力融合细节", GREEN, [
            "road_ad.txt：道路邻接关系",
            "poi_ad.txt：POI 功能相似关系",
            "record_ad.txt：历史碰撞记录关联",
            "三类邻接分别计算注意力权重后进行融合",
        ]),
        (1180, 430, 450, "前端展示输出", RED, [
            "风险热力图",
            "TP / FP / FN 标签校验",
            "区域邻接解释",
            "模型指标与代表性时间帧 JSON",
        ]),
    ]

    for x, width, height, title, color, lines in bottom:
        img, draw = rounded_box(img, (x, detail_y, x + width, detail_y + height), (255, 255, 255), (210, 220, 232), width=2, radius=12, shadow=True)
        draw.text((x + 22, detail_y + 18), title, font=F_HEAD, fill=TEXT)
        draw.line((x + 22, detail_y + 54, x + width - 22, detail_y + 54), fill=color, width=3)
        bullet_lines(draw, x + 22, detail_y + 70, lines, F_BODY, bullet=color, line_h=22, max_width=width - 56)

    # connectors to lower boxes
    arrow(draw, (xs[0] + w * 0.5, y + h + 8), (310, detail_y - 12), BLUE, 2)
    arrow(draw, (xs[2] + w * 0.5, y + h + 8), (860, detail_y - 12), GREEN, 2)
    arrow(draw, (xs[4] + w * 0.5, y + h + 8), (1405, detail_y - 12), RED, 2)

    # footer note
    img, draw = rounded_box(img, (70, 875, 1630, 945), (252, 253, 255), (220, 228, 240), width=1, radius=10, shadow=False)
    footer = (
        "说明：该图按 model.py、train.py 与前端导出脚本的实际流程重绘，"
        "用于论文中展示模型输入、动态演化、空间注意力、时空融合与输出链路。"
    )
    draw.text((92, 895), footer, font=F_FOOT, fill=SUB)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
