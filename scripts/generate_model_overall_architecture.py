from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "model_overall_architecture_flow.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for item in candidates:
        if item and Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


def rounded(draw, xy, fill, outline, radius=22, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, box, text, font, fill="#111827", spacing=7):
    x1, y1, x2, y2 = box
    lines = str(text).split("\n")
    metrics = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [m[2] - m[0] for m in metrics]
    heights = [m[3] - m[1] for m in metrics]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=font, fill=fill)
        y += h + spacing


def arrow(draw, start, end, color="#334155", width=5, head=16):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    p1 = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
    p2 = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
    draw.polygon([end, p1, p2], fill=color)


def connector_label(draw, xy, text, font, fill="#475569"):
    x, y = xy
    pad_x, pad_y = 10, 5
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rounded(draw, (x, y, x + w + pad_x * 2, y + h + pad_y * 2), "#ffffff", "#e2e8f0", radius=10, width=1)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=fill)


def module(draw, box, idx, title, body, fill, outline, title_font, body_font, idx_font):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill, outline, radius=26, width=3)
    rounded(draw, (x1 + 18, y1 + 18, x1 + 64, y1 + 64), "#ffffff", outline, radius=14, width=2)
    centered(draw, (x1 + 18, y1 + 18, x1 + 64, y1 + 64), str(idx), idx_font, outline)
    draw.text((x1 + 82, y1 + 22), title, font=title_font, fill="#0f172a")
    centered(draw, (x1 + 28, y1 + 84, x2 - 28, y2 - 24), body, body_font, "#334155", spacing=9)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (2100, 1250), "#f8fafc")
    d = ImageDraw.Draw(img)

    title_font = load_font(48, True)
    subtitle_font = load_font(25)
    module_title = load_font(28, True)
    body_font = load_font(22)
    small_font = load_font(19)
    idx_font = load_font(25, True)
    note_font = load_font(21)

    navy = "#0f172a"
    muted = "#64748b"
    blue = "#2563eb"
    cyan = "#0891b2"
    green = "#16a34a"
    amber = "#d97706"
    rose = "#e11d48"
    violet = "#7c3aed"

    d.text((90, 58), "模型总体架构流程图", font=title_font, fill=navy)
    d.text(
        (92, 122),
        "3.1 节建议插图：展示交通碰撞风险预测模型从输入构建、动态演化、空间注意力、时空融合到风险输出的五个核心模块",
        font=subtitle_font,
        fill=muted,
    )

    # Main five-module pipeline
    boxes = [
        (90, 270, 430, 590),
        (510, 270, 850, 590),
        (930, 270, 1270, 590),
        (1350, 270, 1690, 590),
        (1770, 270, 2050, 590),
    ]
    modules = [
        (
            "输入构建",
            "交通时序特征 X\n事故标签 y\n阈值信号 threshold_nc\n区域映射 dict_xy\n\n形成滑动窗口：\nT x R x F",
            "#eff6ff",
            "#60a5fa",
        ),
        (
            "动态特征演化",
            "Evolution 层\nupdate-decay 更新\n平滑门控制变化幅度\n\n输出动态状态：\nH_dynamic",
            "#ecfeff",
            "#22d3ee",
        ),
        (
            "多源空间注意力",
            "Road / POI / Record\n三类邻接关系\nScaled Dot Attention\n多轮空间传播\n\n输出空间增强特征",
            "#f0fdf4",
            "#4ade80",
        ),
        (
            "时空融合建模",
            "动态特征 + 静态特征\n拼接后输入 ConvLSTM2D\n学习时间依赖与区域关联\n\n输出融合表示 Z",
            "#fff7ed",
            "#fb923c",
        ),
        (
            "风险预测输出",
            "Dense + Sigmoid\n得到每个区域风险概率\n\n阈值判定 + streaming 后处理\n输出预警结果",
            "#fff1f2",
            "#fb7185",
        ),
    ]

    for idx, (box, (title, body, fill, outline)) in enumerate(zip(boxes, modules), 1):
        module(d, box, idx, title, body, fill, outline, module_title, body_font, idx_font)

    for i in range(len(boxes) - 1):
        x1, y1, x2, y2 = boxes[i]
        nx1, ny1, nx2, ny2 = boxes[i + 1]
        arrow(d, (x2 + 8, (y1 + y2) // 2), (nx1 - 8, (ny1 + ny2) // 2), "#334155", width=5)

    connector_label(d, (442, 228), "滑动窗口序列", small_font)
    connector_label(d, (862, 228), "动态状态 + 原始特征", small_font)
    connector_label(d, (1282, 228), "空间增强表示", small_font)
    connector_label(d, (1702, 228), "融合特征向量", small_font)

    # Supporting inputs under module 3
    support_y = 735
    rounded(d, (650, support_y, 1600, support_y + 250), "#ffffff", "#cbd5e1", radius=28, width=2)
    d.text((690, support_y + 28), "空间关系与注意力融合细节", font=module_title, fill=navy)

    rels = [
        ((700, support_y + 95, 940, support_y + 175), "道路邻接\nroad_ad.txt", "#e0f2fe", cyan),
        ((1005, support_y + 95, 1245, support_y + 175), "功能邻接\npoi_ad.txt", "#dcfce7", green),
        ((1310, support_y + 95, 1550, support_y + 175), "历史碰撞邻接\nrecord_ad.txt", "#fef3c7", amber),
    ]
    for box, text, fill, color in rels:
        rounded(d, box, fill, color, radius=18, width=2)
        centered(d, box, text, body_font, "#1f2937")

    centered(
        d,
        (735, support_y + 190, 1515, support_y + 235),
        "三类邻接分别计算注意力权重，再通过可学习权重融合为区域空间表示",
        note_font,
        muted,
    )
    arrow(d, (820, support_y + 95), (970, 595), cyan, width=4, head=13)
    arrow(d, (1125, support_y + 95), (1100, 595), green, width=4, head=13)
    arrow(d, (1430, support_y + 95), (1230, 595), amber, width=4, head=13)

    # Output products
    rounded(d, (90, 735, 560, 985), "#ffffff", "#cbd5e1", radius=28, width=2)
    d.text((130, 765), "模型训练与评估输出", font=module_title, fill=navy)
    eval_items = [
        ("训练目标", "Focal Loss + 动态差分约束"),
        ("评价指标", "AUC-PR / AUC-ROC / F1 / Recall / Accuracy"),
        ("实验方式", "基线对比 + 消融实验"),
    ]
    y = 825
    for key, value in eval_items:
        d.text((130, y), key, font=note_font, fill=blue)
        d.text((235, y), value, font=note_font, fill="#334155")
        y += 45

    rounded(d, (1650, 735, 2050, 985), "#ffffff", "#cbd5e1", radius=28, width=2)
    d.text((1690, 765), "前端展示输出", font=module_title, fill=navy)
    frontend_items = [
        "风险热力图",
        "TP / FP / FN 标签校验",
        "区域邻接解释",
        "代表性时间帧 JSON",
    ]
    y = 825
    for item in frontend_items:
        d.ellipse((1692, y + 7, 1705, y + 20), fill=rose)
        d.text((1720, y), item, font=note_font, fill="#334155")
        y += 38

    arrow(d, (1910, 590), (1850, 735), rose, width=5)
    connector_label(d, (1595, 715), "部署展示阶段输出前端 JSON", small_font, fill=violet)

    # Caption
    rounded(d, (90, 1060, 2050, 1165), "#ffffff", "#e2e8f0", radius=24, width=2)
    d.text((130, 1090), "图注：", font=module_title, fill=navy)
    d.text(
        (220, 1082),
        "模型首先将交通特征、事故标签、阈值信号和区域邻接关系组织为时序窗口；随后通过动态演化层捕捉风险状态变化，",
        font=note_font,
        fill="#334155",
    )
    d.text(
        (220, 1120),
        "通过多源空间注意力融合道路、POI 与历史碰撞邻接信息，再由 ConvLSTM 完成时空联合建模，最终输出区域碰撞风险概率。",
        font=note_font,
        fill="#334155",
    )

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
