from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "hysteresis_threshold_postprocess_diagram.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for item in candidates:
        if item and Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    lines = []
    current = ""
    for char in text:
        candidate = current + char
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def draw_text_block(draw, box, text, font, fill, line_gap=8, align="center", anchor="middle"):
    x1, y1, x2, y2 = box
    lines = []
    for part in str(text).split("\n"):
        lines.extend(wrap_text(draw, part, font, x2 - x1 - 36))
    heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    if anchor == "top":
        y = y1 + 6
    else:
        y = y1 + (y2 - y1 - total_h) / 2
    for line, h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        if align == "left":
            x = x1 + 24
        else:
            x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + line_gap


def rounded(draw, box, fill, outline, radius=24, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def diamond(draw, center, size, fill, outline, width=3):
    x, y = center
    w, h = size
    points = [(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)]
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=width)
    return points


def arrow(draw, start, end, color="#334155", width=5, head=16):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    p1 = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
    p2 = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
    draw.polygon([end, p1, p2], fill=color)


def label(draw, xy, text, font, fill="#334155", bg="#ffffff", border="#cbd5e1"):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    pad_x, pad_y = 14, 7
    rounded(draw, (x, y, x + bbox[2] + pad_x * 2, y + bbox[3] + pad_y * 2), bg, border, radius=12, width=1)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=fill)


def step_box(draw, box, title, body, fill, outline, title_font, body_font):
    rounded(draw, box, fill, outline, radius=26, width=3)
    x1, y1, x2, y2 = box
    draw.text((x1 + 28, y1 + 20), title, font=title_font, fill="#0f172a")
    draw_text_block(draw, (x1 + 14, y1 + 70, x2 - 14, y2 - 18), body, body_font, "#334155", line_gap=8, anchor="top")


def draw_curve_panel(draw, fonts):
    title_font, body_font, small_font = fonts
    panel = (92, 200, 1710, 465)
    rounded(draw, panel, "#ffffff", "#cbd5e1", radius=30, width=2)
    draw.text((132, 230), "双阈值滞后思想", font=title_font, fill="#0f172a")
    draw_text_block(
        draw,
        (132, 278, 705, 330),
        "风险概率在阈值附近波动时，不立即反复切换预警状态，而是用触发阈值和保持阈值形成缓冲区。",
        small_font,
        "#475569",
        align="left",
        anchor="top",
    )

    chart = (785, 245, 1640, 420)
    x1, y1, x2, y2 = chart
    draw.line((x1, y2, x2, y2), fill="#94a3b8", width=2)
    draw.line((x1, y1, x1, y2), fill="#94a3b8", width=2)
    on_y = y1 + 42
    off_y = y1 + 104
    draw.line((x1, on_y, x2, on_y), fill="#ef4444", width=3)
    draw.line((x1, off_y, x2, off_y), fill="#f59e0b", width=3)
    draw.text((x2 - 170, on_y - 34), "触发阈值 theta_on", font=small_font, fill="#b91c1c")
    draw.text((x2 - 170, off_y + 8), "保持阈值 theta_off", font=small_font, fill="#b45309")

    values = [0.38, 0.51, 0.67, 0.76, 0.71, 0.63, 0.69, 0.58, 0.45]
    pts = []
    for idx, value in enumerate(values):
        x = x1 + 28 + idx * ((x2 - x1 - 64) / (len(values) - 1))
        y = y2 - value * (y2 - y1)
        pts.append((x, y))
    draw.line(pts, fill="#2563eb", width=5, joint="curve")
    for x, y in pts:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2563eb", outline="#ffffff", width=2)

    draw_text_block(
        draw,
        (132, 350, 690, 430),
        "核心作用：减少单阈值附近“刚报警又取消”的震荡，让预警结果更连续，也更符合交通风险处置的业务习惯。",
        body_font,
        "#334155",
        align="left",
    )


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (1800, 1100), "#f8fafc")
    draw = ImageDraw.Draw(img)

    title_font = load_font(50, True)
    subtitle_font = load_font(25)
    box_title = load_font(27, True)
    body_font = load_font(22)
    small_font = load_font(19)
    tiny_font = load_font(17)

    navy = "#0f172a"
    muted = "#64748b"
    blue = "#2563eb"
    cyan = "#0891b2"
    green = "#16a34a"
    amber = "#d97706"
    rose = "#e11d48"
    slate = "#334155"

    draw.text((90, 62), "滞后双阈值后处理流程图", font=title_font, fill=navy)
    draw.text((92, 130), "用于 Attention-LSTM 风险概率输出后的预警状态判定，降低阈值边缘震荡并增强预警连续性。", font=subtitle_font, fill=muted)

    draw_curve_panel(draw, (box_title, body_font, small_font))

    # Main flowchart
    input_box = (95, 565, 385, 725)
    state_center = (555, 645)
    no_box = (790, 510, 1110, 655)
    yes_box = (790, 735, 1110, 880)
    trigger_box = (1250, 490, 1605, 635)
    normal_box = (1250, 675, 1605, 820)
    keep_box = (1250, 860, 1605, 1005)

    step_box(draw, input_box, "输入", "模型输出风险概率 p_t\n读取上一时刻预警状态", "#eff6ff", "#60a5fa", box_title, body_font)
    diamond(draw, state_center, (265, 170), "#f8fafc", "#94a3b8", width=3)
    draw_text_block(draw, (430, 578, 680, 712), "上一时刻\n是否已预警？", box_title, navy, line_gap=10)

    step_box(draw, no_box, "未预警状态", "判断 p_t 是否达到触发阈值\np_t >= theta_on", "#fff7ed", "#fb923c", box_title, body_font)
    step_box(draw, yes_box, "已预警状态", "判断 p_t 是否低于保持阈值\np_t < theta_off", "#ecfeff", "#22d3ee", box_title, body_font)
    step_box(draw, trigger_box, "触发预警", "设置 warning = 1\n前端弹出风险提示卡\n记录预警历史", "#fff1f2", "#fb7185", box_title, body_font)
    step_box(draw, normal_box, "保持正常", "设置 warning = 0\n继续观察后续时间片", "#f1f5f9", "#94a3b8", box_title, body_font)
    step_box(draw, keep_box, "保持或解除", "p_t >= theta_off：保持预警\np_t < theta_off：解除预警", "#f0fdf4", "#4ade80", box_title, body_font)

    arrow(draw, (385, 645), (420, 645), slate)
    label(draw, (390, 598), "状态记忆", tiny_font)

    arrow(draw, (690, 610), (790, 582), amber)
    label(draw, (700, 552), "否", tiny_font, fill=amber)
    arrow(draw, (690, 680), (790, 805), cyan)
    label(draw, (700, 718), "是", tiny_font, fill=cyan)

    arrow(draw, (1110, 570), (1250, 560), rose)
    label(draw, (1140, 520), "达到触发阈值", tiny_font, fill=rose)
    arrow(draw, (1110, 625), (1250, 740), slate)
    label(draw, (1140, 650), "未达到", tiny_font, fill=slate)

    arrow(draw, (1110, 800), (1250, 930), green)
    label(draw, (1130, 856), "未低于保持阈值", tiny_font, fill=green)
    arrow(draw, (1110, 850), (1250, 748), slate)
    label(draw, (1130, 770), "低于保持阈值", tiny_font, fill=slate)

    # Footer notes
    rounded(draw, (95, 1025, 1605, 1082), "#ffffff", "#e2e8f0", radius=20, width=2)
    draw.text((130, 1042), "说明：theta_on 通常高于 theta_off，中间区间作为缓冲带；该策略不改变模型概率本身，只对预警状态进行后处理。", font=body_font, fill="#334155")
    rounded(draw, (1640, 510, 1725, 1005), "#e0f2fe", "#38bdf8", radius=28, width=2)
    draw_text_block(draw, (1640, 510, 1725, 1005), "后处理目标\n\n减少震荡\n保留风险\n提升召回\n便于展示", small_font, "#0f172a", line_gap=12)

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
