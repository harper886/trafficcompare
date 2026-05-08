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

    title = font(48, True)
    subtitle = font(25)
    h2 = font(30, True)
    body = font(23)
    small = font(19)
    formula = font(24)
    note = font(21)

    navy = "#0f172a"
    muted = "#64748b"
    blue = "#2563eb"
    cyan = "#0891b2"
    green = "#16a34a"
    amber = "#d97706"
    rose = "#e11d48"
    violet = "#7c3aed"
    border = "#cbd5e1"

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
        rounded(d, (120, y, 370, y + 76), "#eff6ff", "#93c5fd", radius=18)
        d.text((142, y + 14), label, font=body, fill=blue)
        d.text((142, y + 45), desc, font=small, fill="#334155")
        y += 105

    arrow(d, (410, 540), (510, 540), blue)

    # Encoding core
    rounded(d, (510, 230, 1040, 850), "#ffffff", border, radius=28)
    centered(d, (540, 255, 1010, 305), "正弦 / 余弦位置编码", h2, navy)
    rounded(d, (575, 335, 975, 430), "#ecfeff", "#67e8f9", radius=20)
    centered(d, (575, 335, 975, 430), "PE(pos, 2i) = sin(pos / 10000^(2i/d_model))\nPE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))", formula, cyan)

    d.text((575, 478), "周期性捕捉", font=h2, fill=blue)
    wave(d, 575, 560, 360, 42, blue, phase=0.0, cycles=2.0)
    wave(d, 575, 655, 360, 42, amber, phase=math.pi / 2, cycles=2.0)
    d.line((575, 560, 935, 560), fill="#e2e8f0", width=2)
    d.line((575, 655, 935, 655), fill="#e2e8f0", width=2)
    d.text((950, 538), "sin", font=body, fill=blue)
    d.text((950, 633), "cos", font=body, fill=amber)
    chip(d, 610, 735, "时间周期：23:59 与 00:01 接近", small, "#eff6ff", "#bfdbfe", blue)
    chip(d, 610, 785, "空间环绕：边界附近坐标接近", small, "#fff7ed", "#fed7aa", amber)

    arrow(d, (1040, 540), (1135, 540), green)

    # Fingerprint panel
    rounded(d, (1135, 230, 1785, 850), "#ffffff", border, radius=28)
    centered(d, (1165, 255, 1755, 305), "高维时空指纹", h2, navy)

    # Heatmap-style fingerprint matrix
    x0, y0 = 1210, 355
    cell = 42
    colors = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#fde68a", "#fbbf24", "#dcfce7", "#86efac"]
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
    d.text((1210, 585), "编码向量 E = [sin/cos 时间编码, sin/cos 空间编码, 原始交通特征]", font=note, fill="#334155")

    rounded(d, (1210, 650, 1710, 735), "#f0fdf4", "#86efac", radius=20)
    centered(d, (1210, 650, 1710, 735), "让 LSTM / Attention 同时关注：发生了什么 + 何时发生 + 何地发生", body, green)

    rounded(d, (1210, 765, 1710, 825), "#f5f3ff", "#c4b5fd", radius=18)
    centered(d, (1210, 765, 1710, 825), "学习相对时空关系，弱化对绝对坐标数值的依赖", note, violet)

    # Three benefit cards
    cards = [
        ((90, 910, 560, 1045), "1. 周期性捕捉", "识别时间相邻性与空间边界邻近关系\n避免线性归一化造成语义断裂", "#eff6ff", blue),
        ((715, 910, 1185, 1045), "2. 非线性特征增强", "为每个网格点构造唯一时空指纹\n缓解深层传递中的特征稀释", "#f0fdf4", green),
        ((1330, 910, 1800, 1045), "3. 跨城市迁移", "学习相对位置和周期规律\n为不同城市间迁移提供基础", "#fff7ed", amber),
    ]
    for box, head, text, fill, color in cards:
        rounded(d, box, fill, color, radius=22, width=2)
        d.text((box[0] + 28, box[1] + 22), head, font=body, fill=color)
        lines = text.split("\n")
        for line_idx, line in enumerate(lines):
            d.text((box[0] + 28, box[1] + 64 + line_idx * 30), line, font=small, fill="#334155")

    # Bottom data flow labels
    chip(d, 435, 495, "数值坐标/时间戳", small, "#ffffff", "#e2e8f0", muted)
    chip(d, 1065, 495, "编码后特征向量", small, "#ffffff", "#e2e8f0", muted)

    # Caption
    rounded(d, (80, 1080, 1800, 1145), "#ffffff", "#e2e8f0", radius=20, width=2)
    d.text((115, 1098), "图注：", font=body, fill=navy)
    d.text(
        (190, 1100),
        "时空位置编码通过 sin/cos 周期函数把时间和空间位置映射到高维特征空间，使模型能捕捉周期性、空间邻近性和跨区域迁移规律。",
        font=note,
        fill="#334155",
    )

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
