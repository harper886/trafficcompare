from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "handdrawn-attention-lstm-cn.png"

W, H = 1700, 980
random.seed(7)


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUB = font(16)
F_HEAD = font(20, True)
F_BODY = font(15)
F_SMALL = font(14)

BG = (252, 251, 248)
TEXT = (38, 43, 54)
SUB = (110, 118, 130)
LINE = (96, 110, 132)
BLUE = (74, 124, 214)
GREEN = (73, 159, 110)
ORANGE = (214, 143, 56)
RED = (215, 87, 94)
PURPLE = (130, 107, 214)
CYAN = (70, 160, 188)
GRAY = (240, 242, 245)


def jitter(v, n=2):
    return v + random.randint(-n, n)


def wobble_line(draw, pts, color, width=2, n=2):
    path = [(jitter(x, n), jitter(y, n)) for x, y in pts]
    draw.line(path, fill=color, width=width, joint="curve")


def arrow(draw, start, end, color=LINE, width=2):
    x1, y1 = start
    x2, y2 = end
    wobble_line(draw, [(x1, y1), (x2, y2)], color, width=width, n=1)
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
    draw.polygon([(jitter(x, 1), jitter(y, 1)) for x, y in pts], fill=color)


def rounded_hand_box(img, box, outline, fill=None, width=2, radius=16):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x1, y1, x2, y2 = box
    # sketchy shadow
    for dx, dy, alpha in [(3, 3, 24), (1, 2, 16)]:
        d.rounded_rectangle((x1 + dx, y1 + dy, x2 + dx, y2 + dy), radius=radius, fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, layer.filter(ImageFilter.GaussianBlur(2)))
    d = ImageDraw.Draw(img)
    for off in [0, 1]:
        d.rounded_rectangle((x1 + off, y1 + off, x2 + off, y2 + off), radius=radius, outline=outline, fill=fill, width=width)
    return img, d


def center_text(draw, box, text, fnt, fill=TEXT):
    x1, y1, x2, y2 = box
    tb = draw.textbbox((0, 0), text, font=fnt)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 1), text, font=fnt, fill=fill)


def bullets(draw, x, y, lines, bullet, fnt=F_BODY, color=TEXT, line_h=23):
    cy = y
    for line in lines:
        draw.ellipse((x, cy + 6, x + 6, cy + 12), fill=bullet)
        draw.text((x + 14, cy), line, font=fnt, fill=color)
        cy += line_h
    return cy


def main():
    img = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)

    # subtle background grid and hand-drawn feel
    for gx in range(0, W, 48):
        draw.line((gx, 0, gx, H), fill=(236, 233, 228), width=1)
    for gy in range(0, H, 48):
        draw.line((0, gy, W, gy), fill=(236, 233, 228), width=1)

    draw.text((70, 36), "基于Attention-LSTM的交通碰撞风险预测模型结构图", font=F_TITLE, fill=TEXT)
    draw.text(
        (72, 80),
        "基于统计学、传统机器学习与深度时空表示学习的对比，引出融合时序建模与注意力机制的方法。",
        font=F_SUB,
        fill=SUB,
    )

    # top-left methods strip
    img, draw = rounded_hand_box(img, (62, 132, 1550, 232), outline=PURPLE, fill=(250, 248, 255), width=2, radius=18)
    draw.text((86, 150), "相关技术与方法", font=F_HEAD, fill=TEXT)
    draw.line((86, 182, 260, 182), fill=PURPLE, width=2)
    method_boxes = [
        (90, "统计学方法", "ARIMA\n难以刻画非线性规律", BLUE),
        (330, "传统机器学习", "XGBoost / CatBoost\n难以直接处理拓扑依赖", ORANGE),
        (595, "深度时空学习", "混合架构\n更适合复杂时空依赖", GREEN),
        (870, "本文方法", "Attention + LSTM\n适用于风险预测", RED),
    ]
    for i, (x, title, desc, color) in enumerate(method_boxes):
        img, draw = rounded_hand_box(img, (x, 150, x + 210, 210), outline=color, fill=GRAY, width=2, radius=14)
        center_text(draw, (x + 8, 154, x + 202, 176), title, F_SMALL, color)
        for j, line in enumerate(desc.split("\n")):
            center_text(draw, (x + 12, 177 + j * 13, x + 198, 192 + j * 13), line, font(12), TEXT)
        if i < 3:
            arrow(draw, (x + 214, 180), (x + 228, 180), LINE, 2)

    # main flow boxes
    top_y = 300
    bw, bh = 220, 182
    xs = [72, 372, 672, 972, 1272]
    items = [
        ("1", "输入序列", BLUE, [
            "过去5个时间步",
            "交通时序特征 X",
            "事故标签 y",
            "区域映射 dict_xy",
        ]),
        ("2", "特征编码", PURPLE, [
            "差异特征 DT",
            "静态 / 动态嵌入",
            "时空语义表示",
        ]),
        ("3", "Attention 模块", GREEN, [
            "Query / Key / Value",
            "邻居网格加权",
            "空间注意力聚合",
        ]),
        ("4", "LSTM 编码器", CYAN, [
            "时序状态更新",
            "长短期依赖",
            "隐藏状态输出",
        ]),
        ("5", "风险输出", RED, [
            "Dense + Sigmoid",
            "风险概率 P",
            "高风险预警",
        ]),
    ]

    for i, (num, title, color, lines) in enumerate(items):
        x = xs[i]
        img, draw = rounded_hand_box(img, (x, top_y, x + bw, top_y + bh), outline=color, fill=(255, 255, 255), width=2, radius=18)
        draw.ellipse((x + 16, top_y + 14, x + 40, top_y + 38), outline=color, width=2)
        center_text(draw, (x + 16, top_y + 14, x + 40, top_y + 38), num, F_SMALL, color)
        draw.text((x + 52, top_y + 15), title, font=F_HEAD, fill=TEXT)
        draw.line((x + 16, top_y + 48, x + bw - 16, top_y + 48), fill=color, width=2)
        bullets(draw, x + 18, top_y + 66, lines, color, fnt=F_BODY, line_h=24)
        if i < len(items) - 1:
            arrow(draw, (x + bw + 8, top_y + bh / 2), (xs[i + 1] - 10, top_y + bh / 2), LINE, 2)

    # lower explanation boxes
    lower_y = 545
    lower = [
        (72, 430, "关键点 1：更新 / 衰减", BLUE, [
            "异常时更新特征",
            "正常时指数衰减",
            "减少过平滑",
        ]),
        (520, 560, "关键点 2：三类空间关系", GREEN, [
            "道路邻接 road_ad.txt",
            "POI 相似 poi_ad.txt",
            "历史碰撞 record_ad.txt",
        ]),
        (1130, 420, "关键点 3：输出后处理", RED, [
            "触发阈值 / 解除阈值",
            "减少预警抖动",
            "提高稳定性",
        ]),
    ]

    for x, width, title, color, lines in lower:
        img, draw = rounded_hand_box(img, (x, lower_y, x + width, lower_y + 152), outline=color, fill=(255, 255, 255), width=2, radius=16)
        draw.text((x + 18, lower_y + 14), title, font=F_HEAD, fill=TEXT)
        draw.line((x + 18, lower_y + 46, x + width - 18, lower_y + 46), fill=color, width=2)
        bullets(draw, x + 18, lower_y + 58, lines, color, fnt=F_BODY, line_h=22)

    arrow(draw, (X := 182, top_y + bh + 8), (286, lower_y - 8), BLUE, 2)
    arrow(draw, (682, top_y + bh + 8), (810, lower_y - 8), GREEN, 2)
    arrow(draw, (1272 + bw / 2, top_y + bh + 8), (1350, lower_y - 8), RED, 2)

    # bottom note: two paragraphs
    img, draw = rounded_hand_box(img, (72, 736, 1628, 910), outline=SUB, fill=(255, 255, 255), width=2, radius=18)
    draw.text((94, 754), "图注：", font=F_BODY, fill=TEXT)
    para1 = "模型先将交通时序数据整理为过去5个时间步的输入窗口，并结合事故标签、区域映射和阈值信号形成模型输入。"
    para2 = "随后通过动态演化模块更新区域风险状态，利用道路、POI与历史碰撞记录三类邻接关系进行空间注意力融合，最后通过LSTM完成时序编码并输出风险概率。"
    draw.multiline_text((146, 752), para1, font=F_SMALL, fill=TEXT, spacing=5)
    draw.multiline_text((146, 813), para2, font=F_SMALL, fill=TEXT, spacing=5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
