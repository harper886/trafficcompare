from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "academic-handdrawn-model-architecture.png"

BASE_W, BASE_H = 1700, 980
CANVAS_SCALE = 1.0
FONT_SCALE = 1.48
W, H = int(BASE_W * CANVAS_SCALE), int(BASE_H * CANVAS_SCALE)


def font(size, bold=False):
    paths = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = font(int(38 * FONT_SCALE), True)
F_SUB = font(int(17 * FONT_SCALE))
F_HEAD = font(int(21 * FONT_SCALE), True)
F_BODY = font(int(14 * FONT_SCALE))
F_SMALL = font(int(13 * FONT_SCALE))
F_NOTE = font(int(12 * FONT_SCALE))

BG = (255, 255, 255)
TEXT = (30, 36, 46)
SUB = (104, 112, 123)
GRID = (236, 238, 241)
OUTLINE = (92, 110, 132)
ACCENT = (68, 118, 190)
ACCENT2 = (88, 145, 105)
ACCENT3 = (180, 110, 55)
ACCENT4 = (173, 86, 91)


def text_center(draw, box, text, fnt, fill=TEXT):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 1), text, font=fnt, fill=fill)


def s(v):
    return int(round(v * CANVAS_SCALE))


def sb(box):
    return tuple(s(v) for v in box)


def rect(draw, box, outline=OUTLINE, fill=None, width=2):
    draw.rectangle(box, outline=outline, fill=fill, width=max(1, s(width)))


def arrow(draw, start, end, color=OUTLINE, width=2):
    x1, y1 = s(start[0]), s(start[1])
    x2, y2 = s(end[0]), s(end[1])
    draw.line((x1, y1, x2, y2), fill=color, width=max(1, s(width)))
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 > x1:
            pts = [(x2, y2), (x2 - s(10), y2 - s(5)), (x2 - s(10), y2 + s(5))]
        else:
            pts = [(x2, y2), (x2 + s(10), y2 - s(5)), (x2 + s(10), y2 + s(5))]
    else:
        if y2 > y1:
            pts = [(x2, y2), (x2 - s(5), y2 - s(10)), (x2 + s(5), y2 - s(10))]
        else:
            pts = [(x2, y2), (x2 - s(5), y2 + s(10)), (x2 + s(5), y2 + s(10))]
    draw.polygon(pts, fill=color)


def bullet_block(draw, x, y, lines, fnt, bullet=ACCENT, line_h=22, max_width=0):
    x = s(x)
    cy = s(y)
    for line in lines:
        draw.ellipse((x, cy + s(6), x + s(6), cy + s(12)), fill=bullet)
        draw.text((x + s(14), cy), line, font=fnt, fill=TEXT)
        cy += s(line_h)
    return cy


def add_grid(draw):
    for gx in range(0, W, s(42)):
        draw.line((gx, 0, gx, H), fill=GRID, width=1)
    for gy in range(0, H, s(42)):
        draw.line((0, gy, W, gy), fill=GRID, width=1)


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    add_grid(draw)

    draw.text((s(72), s(34)), "交通碰撞风险预测模型总体架构", font=F_TITLE, fill=TEXT)

    # Main row.
    y = 170
    w = 250
    h = 210
    gap = 34
    xs = [64 + i * (w + gap) for i in range(5)]

    modules = [
        ("1", "输入构建", ACCENT, [
            "交通时序特征 X",
            "事故标签 y",
            "阈值信号 threshold_nc",
            "区域映射 dict_xy",
            "5步滑动窗口",
        ]),
        ("2", "动态特征演化", ACCENT, [
            "Evolution 层",
            "Update / Decay",
            "自适应平滑门",
            "输出动态状态",
        ]),
        ("3", "多源空间注意力", ACCENT2, [
            "Road / POI / Record",
            "三类邻接关系",
            "Dot Attention",
            "邻居特征融合",
        ]),
        ("4", "时空融合建模", ACCENT3, [
            "动态特征 + 静态特征",
            "ConvLSTM2D",
            "时空特征融合",
            "学习风险表示",
        ]),
        ("5", "风险预测输出", ACCENT4, [
            "Dense + Sigmoid",
            "区域风险概率",
            "阈值判定",
            "Streaming 后处理",
        ]),
    ]

    for i, (num, title, color, lines) in enumerate(modules):
        x = xs[i]
        rect(draw, sb((x, y, x + w, y + h)), outline=OUTLINE, fill=None, width=2)
        rect(draw, sb((x, y, x + w, y + 48)), outline=OUTLINE, fill=None, width=1)
        draw.ellipse(sb((x + 12, y + 10, x + 34, y + 32)), outline=color, width=s(2))
        text_center(draw, sb((x + 12, y + 10, x + 34, y + 32)), num, F_SMALL, color)
        draw.text((s(x + 50), s(y + 8)), title, font=F_HEAD, fill=TEXT)
        draw.line(sb((x + 14, y + 52, x + w - 14, y + 52)), fill=color, width=s(2))
        bullet_block(draw, x + 18, y + 66, lines, F_BODY, bullet=color, line_h=27)
        if i < 4:
            arrow(draw, (x + w + 10, y + h / 2), (xs[i + 1] - 10, y + h / 2), OUTLINE, 2)

    # Lower boxes.
    lower_y = 448
    boxes = [
        (70, 350, "模型训练与评估", ACCENT, [
            "训练目标：Focal Loss + 动态约束",
            "评价指标：PR-AUC、ROC-AUC、F1",
            "实验方式：基线对比 + 消融实验",
        ]),
        (470, 560, "空间关系与注意力融合细节", ACCENT2, [
            "road_ad.txt：道路邻接关系",
            "poi_ad.txt：POI 功能相似关系",
            "record_ad.txt：历史碰撞记录关联",
            "三类邻接分别计算权重后融合",
        ]),
        (1070, 480, "前端展示输出", ACCENT4, [
            "风险热力图",
            "TP / FP / FN 标签校验",
            "区域邻接解释",
            "模型指标与 JSON 数据",
        ]),
    ]

    for x, width, title, color, lines in boxes:
        rect(draw, sb((x, lower_y, x + width, lower_y + 175)), outline=OUTLINE, fill=None, width=2)
        rect(draw, sb((x, lower_y, x + width, lower_y + 50)), outline=OUTLINE, fill=None, width=1)
        draw.text((s(x + 18), s(lower_y + 8)), title, font=F_HEAD, fill=TEXT)
        draw.line(sb((x + 16, lower_y + 52, x + width - 16, lower_y + 52)), fill=color, width=s(2))
        bullet_block(draw, x + 16, lower_y + 67, lines, F_SMALL, bullet=color, line_h=23)

    # Connectors.
    arrow(draw, (xs[0] + w / 2, y + h + 6), (280, lower_y - 8), ACCENT, 2)
    arrow(draw, (xs[2] + w / 2, y + h + 6), (830, lower_y - 8), ACCENT2, 2)
    arrow(draw, (xs[4] + w / 2, y + h + 6), (1310, lower_y - 8), ACCENT4, 2)

    # Note box.
    rect(draw, sb((70, 850, 1630, 955)), outline=OUTLINE, fill=None, width=2)
    draw.text((s(90), s(882)), "图注：", font=F_BODY, fill=TEXT)
    para1 = "模型先将交通时序数据整理为过去5个时间步的输入窗口，并结合事故标签、区域映射和阈值信号形成模型输入。"
    para2 = "随后经动态演化、三类邻接注意力融合和 ConvLSTM 时空建模，输出区域风险概率，并用 Streaming 后处理提升预警稳定性。"
    draw.text((s(146), s(879)), para1, font=F_NOTE, fill=TEXT)
    draw.text((s(146), s(906)), para2, font=F_NOTE, fill=TEXT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
