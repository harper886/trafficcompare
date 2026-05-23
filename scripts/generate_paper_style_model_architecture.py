from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "paper-style-model-architecture.png"

W, H = 1600, 900


def font(size, bold=False):
    path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    return ImageFont.truetype(path, size)


F_TITLE = font(42, True)
F_SUB = font(20)
F_HEAD = font(24, True)
F_BODY = font(18)
F_SMALL = font(16)
F_FOOT = font(17)

BG = (248, 250, 253)
TEXT = (28, 39, 55)
MUTED = (94, 110, 130)
BORDER = (145, 165, 190)
BLUE = (55, 123, 210)
BLUE_LIGHT = (232, 242, 255)
GREEN = (64, 150, 105)
GREEN_LIGHT = (235, 249, 240)
ORANGE = (204, 124, 36)
ORANGE_LIGHT = (255, 244, 230)
RED = (205, 76, 87)
RED_LIGHT = (255, 236, 239)
GRAY_LIGHT = (255, 255, 255)


def text_center(draw, box, text, fnt, fill=TEXT):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 1), text, font=fnt, fill=fill)


def draw_multiline(draw, x, y, lines, fnt=F_BODY, fill=TEXT, line_h=29):
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=fnt, fill=fill)


def round_box(draw, box, fill, outline, width=2, r=14):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=(70, 85, 105), width=3):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 14, y2 - 7), (x2 - 14, y2 + 7)]
    else:
        pts = [(x2, y2), (x2 + 14, y2 - 7), (x2 + 14, y2 + 7)]
    draw.polygon(pts, fill=color)


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Title.
    draw.text((70, 42), "交通碰撞风险预测模型总体架构", font=F_TITLE, fill=TEXT)
    draw.text(
        (72, 96),
        "图3-1  展示模型从输入构建、动态演化、空间注意力、时空融合到风险输出的整体流程",
        font=F_SUB,
        fill=MUTED,
    )

    # Main pipeline boxes.
    y = 190
    box_w, box_h = 220, 190
    gap = 44
    xs = [70 + i * (box_w + gap) for i in range(5)]
    modules = [
        {
            "num": "1",
            "title": "输入构建",
            "lines": ["交通时序特征 X", "事故标签 y", "阈值信号 threshold_nc", "区域映射 dict_xy", "形状: T × R × F"],
            "fill": BLUE_LIGHT,
            "line": BLUE,
        },
        {
            "num": "2",
            "title": "动态特征演化",
            "lines": ["Evolution 层", "Update / Decay 更新", "自适应平滑门", "输出动态状态", "H_dynamic"],
            "fill": BLUE_LIGHT,
            "line": BLUE,
        },
        {
            "num": "3",
            "title": "多源空间注意力",
            "lines": ["Road / POI / Record", "三类邻接关系", "Scaled Dot Attention", "邻居特征加权融合", "输出空间增强特征"],
            "fill": GREEN_LIGHT,
            "line": GREEN,
        },
        {
            "num": "4",
            "title": "时空融合建模",
            "lines": ["动态特征 + 静态特征", "拼接后输入 ConvLSTM2D", "学习时间依赖", "与区域空间关联", "输出融合表示 Z"],
            "fill": ORANGE_LIGHT,
            "line": ORANGE,
        },
        {
            "num": "5",
            "title": "风险预测输出",
            "lines": ["Dense + Sigmoid", "得到区域风险概率", "阈值判定", "Streaming 后处理", "输出预警结果"],
            "fill": RED_LIGHT,
            "line": RED,
        },
    ]

    for i, m in enumerate(modules):
        x = xs[i]
        round_box(draw, (x, y, x + box_w, y + box_h), m["fill"], m["line"], width=2, r=15)
        draw.ellipse((x + 16, y + 18, x + 46, y + 48), fill=(255, 255, 255), outline=m["line"], width=2)
        text_center(draw, (x + 16, y + 18, x + 46, y + 48), m["num"], F_SMALL, m["line"])
        draw.text((x + 60, y + 20), m["title"], font=F_HEAD, fill=TEXT)
        draw.line((x + 20, y + 62, x + box_w - 20, y + 62), fill=m["line"], width=2)
        draw_multiline(draw, x + 30, y + 82, m["lines"], fnt=F_SMALL, fill=TEXT, line_h=24)
        if i < 4:
            arrow(draw, (x + box_w + 7, y + box_h // 2), (xs[i + 1] - 10, y + box_h // 2))

    # Lower detail boxes.
    lower_y = 455
    lower_boxes = [
        (70, lower_y, 360, 158, "模型训练与评估", ["训练目标：Focal Loss + 动态约束", "评价指标：AUC-PR / AUC-ROC", "F1 / Recall / Accuracy", "实验方式：模型对比 + 消融实验"], BLUE),
        (470, lower_y, 500, 158, "空间关系与注意力融合细节", ["road_ad.txt：道路邻接关系", "poi_ad.txt：POI功能相似关系", "record_ad.txt：历史碰撞记录关联", "三类邻接分别计算注意力权重，再融合为空间增强表示"], GREEN),
        (1010, lower_y, 360, 158, "前端展示输出", ["风险热力图", "TP / FP / FN 标签校验", "区域邻接解释", "代表性时间帧 JSON", "模型指标展示"], RED),
    ]

    for x, yy, ww, hh, title, lines, color in lower_boxes:
        round_box(draw, (x, yy, x + ww, yy + hh), GRAY_LIGHT, (205, 215, 228), width=2, r=14)
        draw.text((x + 24, yy + 20), title, font=F_HEAD, fill=TEXT)
        draw.line((x + 24, yy + 56, x + ww - 24, yy + 56), fill=color, width=3)
        bullet_y = yy + 72
        for idx, line in enumerate(lines):
            by = bullet_y + idx * 22
            draw.ellipse((x + 26, by + 7, x + 34, by + 15), fill=color)
            draw.text((x + 44, by), line, font=F_SMALL, fill=TEXT)

    # Connector lines to lower boxes.
    arrow(draw, (xs[0] + box_w / 2, y + box_h + 8), (250, lower_y - 10), BLUE, width=2)
    arrow(draw, (xs[2] + box_w / 2, y + box_h + 8), (720, lower_y - 10), GREEN, width=2)
    arrow(draw, (xs[4] + box_w / 2, y + box_h + 8), (1190, lower_y - 10), RED, width=2)

    # Note box.
    note_y = 690
    round_box(draw, (70, note_y, 1530, note_y + 110), (255, 255, 255), (210, 220, 232), width=2, r=14)
    draw.text((96, note_y + 24), "图注：", font=F_FOOT, fill=TEXT)
    note = (
        "模型首先将交通时序数据组织为过去5个时间步的历史窗口；随后通过动态演化模块更新区域风险状态，"
        "利用道路、POI与历史碰撞记录三类邻接关系进行空间注意力融合；最后通过ConvLSTM完成时空建模，"
        "输出区域碰撞风险概率，并通过Streaming后处理提升预警结果稳定性。"
    )
    # Manual line wrap for Chinese.
    lines = [
        note[:48],
        note[48:96],
        note[96:],
    ]
    draw_multiline(draw, 160, note_y + 22, lines, fnt=F_FOOT, fill=TEXT, line_h=28)

    # Small footer.
    draw.text((70, 840), "注：该图根据 model.py、train.py 与前端导出流程整理绘制。", font=F_SMALL, fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
