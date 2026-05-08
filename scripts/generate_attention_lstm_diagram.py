from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "attention_lstm_fusion_diagram.png"


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rounded(draw, box, fill, outline="#cbd5e1", radius=24, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, box, text, fnt, fill="#0f172a", spacing=8):
    x1, y1, x2, y2 = box
    lines = str(text).split("\n")
    metrics = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [m[2] - m[0] for m in metrics]
    heights = [m[3] - m[1] for m in metrics]
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


def label(draw, xy, text, fnt, fill="#475569"):
    draw.text(xy, text, font=fnt, fill=fill)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (1800, 1100), "#f8fafc")
    d = ImageDraw.Draw(img)

    title = font(46, True)
    subtitle = font(24)
    h2 = font(30, True)
    body = font(24)
    small = font(19)
    formula = font(22)
    badge = font(22, True)

    navy = "#0f172a"
    muted = "#64748b"
    blue = "#2563eb"
    cyan = "#0891b2"
    green = "#16a34a"
    amber = "#d97706"
    rose = "#e11d48"
    border = "#cbd5e1"

    d.text((80, 52), "Attention-LSTM 单元结构与注意力融合位置", font=title, fill=navy)
    d.text(
        (82, 112),
        "用于补充 2.1.1 Attention-LSTM 技术：展示 LSTM 门控更新、隐藏状态序列和 Attention 上下文向量的融合路径",
        font=subtitle,
        fill=muted,
    )

    # Input sequence
    rounded(d, (80, 205, 420, 850), "#ffffff", border)
    centered(d, (105, 225, 395, 270), "输入时序特征", h2, navy)
    inputs = [
        ("x(t-4)", "历史交通流 / 阈值 / 区域特征"),
        ("x(t-3)", "多时间片输入"),
        ("x(t-2)", "time x region x feature"),
        ("x(t-1)", "近邻风险变化"),
        ("x(t)", "当前时间片"),
    ]
    y = 315
    for name, desc in inputs:
        rounded(d, (135, y, 365, y + 72), "#eff6ff", "#93c5fd", radius=18)
        centered(d, (150, y + 8, 220, y + 64), name, body, blue)
        d.text((220, y + 16), desc, font=small, fill="#334155")
        y += 95
    arrow(d, (420, 520), (520, 520), blue)

    # LSTM unit
    rounded(d, (520, 205, 1040, 850), "#ffffff", border)
    centered(d, (545, 225, 1015, 270), "典型 LSTM 单元", h2, navy)

    # Cell state line
    arrow(d, (560, 330), (1000, 330), green, width=6)
    label(d, (565, 292), "C(t-1)", formula, green)
    label(d, (943, 292), "C(t)", formula, green)
    label(d, (705, 292), "细胞状态主通道", small, muted)

    gates = [
        ((575, 430, 710, 520), "遗忘门\nf(t)", "#dcfce7", green),
        ((735, 430, 870, 520), "输入门\ni(t)", "#dbeafe", blue),
        ((895, 430, 1015, 520), "候选状态\nC_hat(t)", "#fef3c7", amber),
        ((670, 615, 830, 705), "输出门\no(t)", "#fce7f3", rose),
    ]
    for box, text, fill, color in gates:
        rounded(d, box, fill, color, radius=18, width=2)
        centered(d, box, text, small, "#1f2937", spacing=5)

    # Update equations
    rounded(d, (575, 545, 1015, 590), "#f8fafc", border, radius=14)
    centered(d, (575, 545, 1015, 590), "C(t) = f(t) * C(t-1) + i(t) * C_hat(t)", formula, green)
    rounded(d, (575, 730, 1015, 775), "#f8fafc", border, radius=14)
    centered(d, (575, 730, 1015, 775), "h(t) = o(t) * tanh(C(t))", formula, rose)

    arrow(d, (780, 775), (780, 845), rose)
    label(d, (805, 807), "输出当前隐藏状态 h(t)", body, rose)

    # Hidden sequence to attention
    arrow(d, (1040, 520), (1130, 520), blue)
    rounded(d, (1130, 205, 1660, 850), "#ffffff", border)
    centered(d, (1160, 225, 1630, 270), "Attention 融合模块", h2, navy)

    rounded(d, (1190, 320, 1600, 390), "#ecfeff", "#67e8f9", radius=18)
    centered(d, (1190, 320, 1600, 390), "隐藏状态序列 H = [h(t-4), h(t-3), ..., h(t)]", formula, cyan)

    rounded(d, (1190, 430, 1600, 505), "#eff6ff", "#93c5fd", radius=18)
    centered(d, (1190, 430, 1600, 505), "相关性打分：e_i = score(h_i, h(t))", formula, blue)

    rounded(d, (1190, 545, 1600, 620), "#fef3c7", "#f59e0b", radius=18)
    centered(d, (1190, 545, 1600, 620), "注意力权重：alpha_i = softmax(e_i)", formula, amber)

    rounded(d, (1190, 660, 1600, 735), "#dcfce7", "#86efac", radius=18)
    centered(d, (1190, 660, 1600, 735), "上下文向量：c(t) = sum(alpha_i * h_i)", formula, green)

    arrow(d, (1395, 390), (1395, 430), cyan, width=4)
    arrow(d, (1395, 505), (1395, 545), blue, width=4)
    arrow(d, (1395, 620), (1395, 660), amber, width=4)

    # Fusion and prediction
    rounded(d, (705, 900, 1445, 1010), "#fff7ed", "#fb923c", radius=26, width=3)
    centered(d, (725, 918, 1015, 990), "Attention 融合位置", badge, amber)
    centered(d, (1010, 910, 1425, 965), "z(t) = concat(h(t), c(t))", formula, navy)
    centered(d, (1010, 960, 1425, 995), "Dense / Sigmoid → 区域碰撞风险概率", body, rose)

    arrow(d, (780, 850), (920, 900), rose, width=5)
    arrow(d, (1395, 735), (1260, 900), green, width=5)
    arrow(d, (1445, 955), (1595, 955), rose, width=5)
    rounded(d, (1595, 900, 1715, 1010), "#ffe4e6", "#fb7185", radius=24, width=2)
    centered(d, (1595, 900, 1715, 1010), "y_hat(t)\n风险输出", body, rose)

    # Callout
    rounded(d, (1065, 875, 1365, 930), "#fff1f2", "#fb7185", radius=18, width=2)
    centered(d, (1065, 875, 1365, 930), "在 LSTM 输出后引入 Attention，突出关键时间片/区域特征", small, rose)

    # Legend
    rounded(d, (80, 910, 610, 1010), "#ffffff", border, radius=20)
    d.text((110, 932), "图注：", font=badge, fill=navy)
    d.text((180, 932), "LSTM 负责建模交通风险的时序依赖；Attention 根据隐藏状态重要性生成上下文向量，", font=small, fill="#334155")
    d.text((180, 964), "再与当前隐藏状态拼接融合，用于最终风险预测。", font=small, fill="#334155")

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
