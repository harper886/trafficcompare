from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "spatio_temporal_encoding_diagram.png"


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded(draw, box, fill, outline, radius=22, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, box, text, fnt, fill="#0f172a", spacing=7):
    x1, y1, x2, y2 = box
    lines = str(text).split("\n")
    boxes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + spacing


def draw_lines(draw, xy, lines, fnt, fill="#334155", line_h=28):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h


def arrow(draw, start, end, color="#334155", width=5, head=16):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    p1 = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
    p2 = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
    draw.polygon([end, p1, p2], fill=color)


def wave(draw, x0, y0, width, amp, color, phase=0.0, cycles=2.0, line_width=4):
    points = []
    for i in range(width + 1):
        x = x0 + i
        y = y0 + math.sin((i / width) * cycles * 2 * math.pi + phase) * amp
        points.append((x, y))
    draw.line(points, fill=color, width=line_width)


def chip(draw, x, y, text, fnt, fill, outline, text_color="#0f172a"):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rounded(draw, (x, y, x + w + 26, y + h + 16), fill, outline, radius=14, width=2)
    draw.text((x + 13, y + 7), text, font=fnt, fill=text_color)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (1900, 1180), "#f8fafc")
    d = ImageDraw.Draw(img)

    title = font(54, True)
    subtitle = font(28)
    h2 = font(35, True)
    body = font(27)
    small = font(22)
    formula = font(23)
    note = font(24)

    navy = "#172033"
    muted = "#687488"
    blue = "#3f6f9f"
    cyan = "#4f8a94"
    green = "#547d5f"
    amber = "#9b6a3c"
    rose = "#9b5960"
    violet = "#75619a"
    border = "#c5ceda"

    d.text((80, 52), "时空位置编码（Spatio-Temporal Encoding）技术解释图", font=title, fill=navy)
    d.text(
        (82, 116),
        "将经纬度与时间戳从原始数值映射为周期感知的高维向量，帮助模型理解“何时、何地”发生交通风险",
        font=subtitle,
        fill=muted,
    )

    # Raw input panel
    rounded(d, (80, 230, 410, 850), "#ffffff", border, radius=28)
    centered(d, (110, 255, 380, 305), "原始输入", h2, navy)
    raw_items = [
        ("经度 lon", "例如 179.9E / -179.9W"),
        ("纬度 lat", "城市网格空间位置"),
        ("时间 t", "小时 / 星期 / 日期"),
        ("交通特征 X", "流量、阈值、历史事故"),
    ]
    y = 350
    for label, desc in raw_items:
        rounded(d, (120, y, 378, y + 80), "#edf3f8", "#a9bacd", radius=18)
        d.text((142, y + 14), label, font=body, fill=blue)
        d.text((142, y + 48), desc, font=small, fill="#405064")
        y += 105

    arrow(d, (410, 540), (510, 540), blue)

    # Encoding core
    rounded(d, (510, 230, 1040, 850), "#ffffff", border, radius=28)
    centered(d, (540, 255, 1010, 305), "正弦 / 余弦位置编码", h2, navy)
    rounded(d, (570, 328, 982, 435), "#eef5f6", "#9bbbc1", radius=20)
    centered(
        d,
        (570, 328, 982, 435),
        "PE(pos, 2i) = sin(pos / 10000^(2i/d))\nPE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
        formula,
        cyan,
        spacing=10,
    )

    d.text((600, 470), "周期性捕捉", font=h2, fill=blue)
    wave(d, 585, 565, 345, 38, blue, phase=0.0, cycles=2.0)
    wave(d, 585, 655, 345, 38, amber, phase=math.pi / 2, cycles=2.0)
    d.line((585, 565, 930, 565), fill="#e2e8f0", width=2)
    d.line((585, 655, 930, 655), fill="#e2e8f0", width=2)
    d.text((944, 543), "sin", font=body, fill=blue)
    d.text((944, 633), "cos", font=body, fill=amber)
    chip(d, 595, 735, "时间周期：23:59 与 00:01 接近", small, "#edf3f8", "#b9c8d7", blue)
    chip(d, 595, 790, "空间环绕：边界附近坐标接近", small, "#f7f0e8", "#d7b999", amber)

    arrow(d, (1040, 540), (1135, 540), green)

    # Fingerprint panel
    rounded(d, (1135, 230, 1785, 850), "#ffffff", border, radius=28)
    centered(d, (1165, 255, 1755, 305), "高维时空指纹", h2, navy)

    # Heatmap-style fingerprint matrix
    x0, y0 = 1210, 355
    cell = 42
    colors = ["#dce6ef", "#c8d8e8", "#aebfce", "#8ea7bd", "#ead9b5", "#d1b37b", "#dce9dc", "#aac7ad"]
    for r in range(5):
        for c in range(8):
            color = colors[(r * 3 + c * 2) % len(colors)]
            d.rounded_rectangle(
                (x0 + c * cell, y0 + r * cell, x0 + c * cell + 34, y0 + r * cell + 34),
                radius=7,
                fill=color,
                outline="#ffffff",
                width=2,
            )
    centered(
        d,
        (1210, 570, 1710, 628),
        "编码向量 E = [时间编码, 空间编码, 交通特征]",
        note,
        "#405064",
    )

    rounded(d, (1210, 645, 1710, 745), "#eef5ef", "#adc7b1", radius=20)
    centered(d, (1210, 645, 1710, 745), "让 LSTM / Attention 同时关注：\n发生了什么 + 何时发生 + 何地发生", body, green)

    rounded(d, (1210, 765, 1710, 835), "#f2f0f7", "#beb4d0", radius=18)
    centered(d, (1210, 765, 1710, 835), "学习相对时空关系\n弱化对绝对坐标数值的依赖", note, violet)

    # Three benefit cards
    cards = [
        ((90, 910, 560, 1045), "1. 周期性捕捉", "识别时间相邻性与空间边界邻近关系\n避免线性归一化造成语义断裂", "#edf3f8", blue),
        ((715, 910, 1185, 1045), "2. 非线性特征增强", "为每个网格点构造唯一时空指纹\n缓解深层传递中的特征稀释", "#eef5ef", green),
        ((1330, 910, 1800, 1045), "3. 跨城市迁移", "学习相对位置和周期规律\n为不同城市间迁移提供基础", "#f7f0e8", amber),
    ]
    for box, head, text, fill, color in cards:
        rounded(d, box, fill, color, radius=22, width=2)
        d.text((box[0] + 28, box[1] + 22), head, font=body, fill=color)
        lines = text.split("\n")
        for line_idx, line in enumerate(lines):
            d.text((box[0] + 28, box[1] + 66 + line_idx * 34), line, font=small, fill="#405064")

    # Bottom data flow labels
    chip(d, 430, 500, "数值坐标/时间戳", font(18), "#ffffff", "#e2e8f0", muted)
    chip(d, 1068, 500, "编码后特征向量", font(18), "#ffffff", "#e2e8f0", muted)

    # Caption
    rounded(d, (80, 1080, 1800, 1145), "#ffffff", "#e2e8f0", radius=20, width=2)
    d.text((115, 1098), "图注：", font=body, fill=navy)
    d.text(
        (190, 1100),
        "时空位置编码通过 sin/cos 周期函数把时间和空间位置映射到高维特征空间，使模型能捕捉周期性、空间邻近性和跨区域迁移规律。",
        font=note,
        fill="#405064",
    )

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
