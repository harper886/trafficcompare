from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "chapter3_overall_project_diagram.png"


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


def rounded(draw, box, fill, outline, radius=24, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw, box, text, font, fill="#0f172a", spacing=8):
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


def arrow(draw, start, end, color="#334155", width=5, head=17):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    p1 = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
    p2 = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
    draw.polygon([end, p1, p2], fill=color)


def label(draw, x, y, text, font, fill="#475569"):
    pad_x, pad_y = 12, 6
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rounded(draw, (x, y, x + w + pad_x * 2, y + h + pad_y * 2), "#ffffff", "#e2e8f0", radius=12, width=1)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=fill)


def module(draw, box, number, title, body, fill, outline, fonts):
    x1, y1, x2, y2 = box
    title_font, body_font, num_font = fonts
    rounded(draw, box, fill, outline, radius=28, width=3)
    rounded(draw, (x1 + 22, y1 + 22, x1 + 72, y1 + 72), "#ffffff", outline, radius=16, width=2)
    text_center(draw, (x1 + 22, y1 + 22, x1 + 72, y1 + 72), str(number), num_font, outline)
    draw.text((x1 + 88, y1 + 26), title, font=title_font, fill="#0f172a")
    text_center(draw, (x1 + 28, y1 + 88, x2 - 28, y2 - 24), body, body_font, "#334155", spacing=8)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (2200, 1380), "#f8fafc")
    d = ImageDraw.Draw(img)

    title_font = load_font(50, True)
    subtitle_font = load_font(25)
    module_title = load_font(28, True)
    body_font = load_font(22)
    small_font = load_font(19)
    note_font = load_font(21)
    num_font = load_font(26, True)

    navy = "#0f172a"
    muted = "#64748b"
    blue = "#2563eb"
    cyan = "#0891b2"
    green = "#16a34a"
    amber = "#d97706"
    rose = "#e11d48"
    violet = "#7c3aed"

    d.text((90, 58), "交通碰撞风险预测项目整体技术路线图", font=title_font, fill=navy)
    d.text(
        (92, 124),
        "面向第三章方法设计：从双城数据输入、时空信息编码、多源空间注意力、时序建模到风险输出与前端可视化",
        font=subtitle_font,
        fill=muted,
    )

    # Main pipeline
    y0, h = 250, 300
    boxes = [
        (80, y0, 390, y0 + h),
        (470, y0, 780, y0 + h),
        (860, y0, 1170, y0 + h),
        (1250, y0, 1560, y0 + h),
        (1640, y0, 1950, y0 + h),
    ]
    modules = [
        ("数据输入构建", "NYC / Chicago 双城数据\n交通时序特征 X\n事故标签 y\n阈值信号 threshold_nc\n区域映射 dict_xy", "#eff6ff", "#60a5fa"),
        ("时空信息编码", "经纬度位置编码\nPE(lat), PE(lon)\n相对时间编码\n融合交通状态特征\n形成高维时空向量", "#ecfeff", "#22d3ee"),
        ("动态特征演化", "Evolution 模块\nupdate-decay 更新\n平滑门控制变化幅度\n捕捉风险状态随时间变化", "#f0fdf4", "#4ade80"),
        ("多源空间注意力", "Road / POI / Record\n三类邻接关系\nScaled Dot Attention\n学习邻居区域贡献权重", "#fff7ed", "#fb923c"),
        ("风险预测输出", "ConvLSTM2D 时空融合\nDense + Sigmoid\n输出区域碰撞风险概率\n阈值判断与流式后处理", "#fff1f2", "#fb7185"),
    ]
    fonts = (module_title, body_font, num_font)
    for idx, (box, item) in enumerate(zip(boxes, modules), 1):
        module(d, box, idx, item[0], item[1], item[2], item[3], fonts)

    for i in range(len(boxes) - 1):
        x1, y1, x2, y2 = boxes[i]
        nx1, ny1, nx2, ny2 = boxes[i + 1]
        arrow(d, (x2 + 8, (y1 + y2) // 2), (nx1 - 8, (ny1 + ny2) // 2))

    label(d, 405, 202, "滑动时间窗口 T=5", small_font)
    label(d, 795, 202, "编码后的时空特征", small_font)
    label(d, 1185, 202, "动态隐藏状态 H", small_font)
    label(d, 1575, 202, "空间增强表示 Z", small_font)

    # Detail panel for section 3.2
    detail_y = 640
    rounded(d, (80, detail_y, 1950, detail_y + 320), "#ffffff", "#cbd5e1", radius=30, width=2)
    d.text((120, detail_y + 30), "3.2 时空信息编码机制细节", font=module_title, fill=navy)
    d.text(
        (120, detail_y + 78),
        "利用正余弦位置编码将球面经纬度与相对时间映射到连续高维空间，缓解绝对坐标语义丢失、边界邻近断裂和虚假周期关联问题。",
        font=note_font,
        fill="#475569",
    )

    detail_boxes = [
        ((125, detail_y + 145, 525, detail_y + 270), "空间编码", "PE(lat, 2k)=sin(lat/10000^(2k/D))\nPE(lat, 2k+1)=cos(lat/10000^(2k/D))\n经度 lon 执行相同编码后拼接", "#eff6ff", blue),
        ((610, detail_y + 145, 1010, detail_y + 270), "相对时间编码", "使用事件间隔与历史窗口位置\n表达“降雨后两小时”等相对关系\n减少绝对时间带来的虚假关联", "#ecfeff", cyan),
        ((1095, detail_y + 145, 1495, detail_y + 270), "特征融合", "将空间编码、时间编码与交通特征拼接\n构造 time × region × feature 输入张量\n作为后续模型统一输入", "#f0fdf4", green),
        ((1580, detail_y + 145, 1900, detail_y + 270), "建模作用", "增强周期性表达\n保持空间邻近关系\n支持跨区域风险迁移分析", "#fff7ed", amber),
    ]
    for box, head, body, fill, color in detail_boxes:
        rounded(d, box, fill, color, radius=22, width=2)
        d.text((box[0] + 24, box[1] + 18), head, font=body_font, fill=color)
        text_center(d, (box[0] + 20, box[1] + 50, box[2] - 20, box[3] - 10), body, small_font, "#334155", spacing=5)

    for a, b in [(525, 610), (1010, 1095), (1495, 1580)]:
        arrow(d, (a + 8, detail_y + 207), (b - 8, detail_y + 207), "#94a3b8", width=4, head=13)

    # Inputs and outputs
    bottom_y = 1040
    rounded(d, (80, bottom_y, 610, bottom_y + 210), "#ffffff", "#cbd5e1", radius=28, width=2)
    d.text((120, bottom_y + 28), "输入数据文件", font=module_title, fill=navy)
    input_items = [
        "data_nyc.npy / data_chicago.npy：交通时序特征",
        "label.npy：事故标签",
        "road_ad.txt / poi_ad.txt / record_ad.txt：空间邻接关系",
        "dict_xy.npy：区域坐标映射",
    ]
    for i, item in enumerate(input_items):
        d.ellipse((125, bottom_y + 86 + i * 36, 137, bottom_y + 98 + i * 36), fill=blue)
        d.text((150, bottom_y + 78 + i * 36), item, font=small_font, fill="#334155")

    rounded(d, (760, bottom_y, 1360, bottom_y + 210), "#ffffff", "#cbd5e1", radius=28, width=2)
    d.text((800, bottom_y + 28), "训练与评价", font=module_title, fill=navy)
    train_items = [
        "训练入口：train.py",
        "模型结构：model.py 中的 MYPLAN、Evolution、MultiAttention",
        "评价指标：AUC-PR、AUC-ROC、F1、Recall、Accuracy",
        "实验方式：基线对比与消融实验",
    ]
    for i, item in enumerate(train_items):
        d.ellipse((805, bottom_y + 86 + i * 36, 817, bottom_y + 98 + i * 36), fill=green)
        d.text((830, bottom_y + 78 + i * 36), item, font=small_font, fill="#334155")

    rounded(d, (1510, bottom_y, 1950, bottom_y + 210), "#ffffff", "#cbd5e1", radius=28, width=2)
    d.text((1550, bottom_y + 28), "展示与交付", font=module_title, fill=navy)
    output_items = [
        "frontend_predictions_*.json：预测结果",
        "frontend_metrics.json：模型性能指标",
        "frontend_topology.json：空间拓扑关系",
        "dashboard_fixed.html：可视化系统页面",
    ]
    for i, item in enumerate(output_items):
        d.ellipse((1555, bottom_y + 86 + i * 36, 1567, bottom_y + 98 + i * 36), fill=rose)
        d.text((1580, bottom_y + 78 + i * 36), item, font=small_font, fill="#334155")

    arrow(d, (610, bottom_y + 105), (760, bottom_y + 105), "#94a3b8", width=4, head=13)
    arrow(d, (1360, bottom_y + 105), (1510, bottom_y + 105), "#94a3b8", width=4, head=13)

    # Caption-like note inside the image
    rounded(d, (80, 1280, 1950, 1340), "#ffffff", "#e2e8f0", radius=20, width=2)
    d.text((115, 1298), "图注：", font=note_font, fill=navy)
    d.text(
        (185, 1298),
        "该技术路线图突出 3.2 节时空信息编码在整体模型中的位置：编码结果作为动态演化、空间注意力和时空融合模块的统一输入基础。",
        font=note_font,
        fill="#334155",
    )

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
