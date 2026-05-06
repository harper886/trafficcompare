from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "project_flowchart.png"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def rounded_box(draw, xy, fill, outline, radius=18, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def multiline_center(draw, box, text, font, fill, spacing=8):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2 - 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x1 + ((x2 - x1) - w) / 2, y), line, font=font, fill=fill)
        y += h + spacing


def arrow(draw, start, end, color="#394150", width=4):
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        points = [(ex, ey), (ex - sign * 14, ey - 8), (ex - sign * 14, ey + 8)]
    else:
        sign = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 8, ey - sign * 14), (ex + 8, ey - sign * 14)]
    draw.polygon(points, fill=color)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1800, 1200
    img = Image.new("RGB", (width, height), "#f7f9fc")
    draw = ImageDraw.Draw(img)

    title_font = load_font(46)
    subtitle_font = load_font(25)
    node_font = load_font(26)
    small_font = load_font(22)
    badge_font = load_font(23)

    draw.text((80, 50), "交通碰撞风险预测与可解释可视化系统流程图", font=title_font, fill="#18202f")
    draw.text(
        (82, 112),
        "基于 NYC / Chicago 双城市数据，完成数据构建、Attention-LSTM 训练、结果导出与前端可视化展示",
        font=subtitle_font,
        fill="#526071",
    )

    columns = [
        ("数据准备", "#e9f2ff", "#4078c0"),
        ("模型训练", "#edf7ee", "#3f8f4e"),
        ("结果导出", "#fff3df", "#c67d18"),
        ("可视化展示", "#f1edff", "#7551c9"),
        ("论文答辩", "#ffeef0", "#c8475b"),
    ]
    col_x = [80, 430, 780, 1130, 1480]
    col_w = 260
    y_top = 245
    row_gap = 170
    box_h = 108

    for i, (name, fill, outline) in enumerate(columns):
        x = col_x[i]
        rounded_box(draw, (x, y_top - 58, x + col_w, y_top - 10), fill, outline, radius=12, width=2)
        multiline_center(draw, (x, y_top - 58, x + col_w, y_top - 10), name, badge_font, outline, spacing=0)

    nodes = {
        "raw": ((80, y_top, 340, y_top + box_h), "原始交通数据\nNYC / Chicago", "#e9f2ff", "#4078c0"),
        "prep": ((80, y_top + row_gap, 340, y_top + row_gap + box_h), "数据预处理\n时序特征 + 标签", "#e9f2ff", "#4078c0"),
        "spatial": ((80, y_top + row_gap * 2, 340, y_top + row_gap * 2 + box_h), "空间关系构建\n道路 / POI / 历史邻接", "#e9f2ff", "#4078c0"),
        "train": ((430, y_top + 70, 690, y_top + 70 + box_h), "train.py\n模型训练入口", "#edf7ee", "#3f8f4e"),
        "model": ((430, y_top + row_gap + 70, 690, y_top + row_gap + 70 + box_h), "Attention-LSTM\n风险动态建模", "#edf7ee", "#3f8f4e"),
        "eval": ((430, y_top + row_gap * 2 + 70, 690, y_top + row_gap * 2 + 70 + box_h), "实验评估\nAUC / F1 / Recall", "#edf7ee", "#3f8f4e"),
        "infer": ((780, y_top, 1040, y_top + box_h), "推理与后处理\n平滑 + 流式输出", "#fff3df", "#c67d18"),
        "json": ((780, y_top + row_gap, 1040, y_top + row_gap + box_h), "前端 JSON 导出\n预测 / 指标 / 拓扑", "#fff3df", "#c67d18"),
        "assets": ((780, y_top + row_gap * 2, 1040, y_top + row_gap * 2 + box_h), "结果文件\nresults / outputs", "#fff3df", "#c67d18"),
        "dash": ((1130, y_top + 70, 1390, y_top + 70 + box_h), "dashboard_fixed.html\n交互式大屏", "#f1edff", "#7551c9"),
        "views": ((1130, y_top + row_gap + 70, 1390, y_top + row_gap + 70 + box_h), "风险热力图\n时间轴演化", "#f1edff", "#7551c9"),
        "explain": ((1130, y_top + row_gap * 2 + 70, 1390, y_top + row_gap * 2 + 70 + box_h), "TP / FP / FN 校验\n局部空间解释", "#f1edff", "#7551c9"),
        "paper": ((1480, y_top, 1740, y_top + box_h), "论文正文\n方法 + 实验 + 系统", "#ffeef0", "#c8475b"),
        "ppt": ((1480, y_top + row_gap, 1740, y_top + row_gap + box_h), "答辩 PPT\n讲稿 + 问答", "#ffeef0", "#c8475b"),
        "demo": ((1480, y_top + row_gap * 2, 1740, y_top + row_gap * 2 + box_h), "现场演示\n运行与展示脚本", "#ffeef0", "#c8475b"),
    }

    for box, text, fill, outline in nodes.values():
        rounded_box(draw, box, fill, outline)
        multiline_center(draw, box, text, node_font, "#1f2937")

    def right_mid(key):
        x1, y1, x2, y2 = nodes[key][0]
        return x2, (y1 + y2) // 2

    def left_mid(key):
        x1, y1, x2, y2 = nodes[key][0]
        return x1, (y1 + y2) // 2

    def bottom_mid(key):
        x1, y1, x2, y2 = nodes[key][0]
        return (x1 + x2) // 2, y2

    def top_mid(key):
        x1, y1, x2, y2 = nodes[key][0]
        return (x1 + x2) // 2, y1

    for a, b in [("raw", "prep"), ("prep", "spatial"), ("train", "model"), ("model", "eval"), ("infer", "json"), ("json", "assets"), ("dash", "views"), ("views", "explain"), ("paper", "ppt"), ("ppt", "demo")]:
        arrow(draw, bottom_mid(a), top_mid(b))

    for a, b in [("prep", "train"), ("spatial", "train"), ("model", "infer"), ("eval", "json"), ("assets", "dash"), ("views", "paper"), ("explain", "ppt"), ("dash", "demo")]:
        arrow(draw, right_mid(a), left_mid(b))

    legend_y = 1010
    rounded_box(draw, (80, legend_y, 1740, 1125), "#ffffff", "#d4dbe7", radius=18, width=2)
    draw.text((116, legend_y + 28), "核心交付：", font=small_font, fill="#18202f")
    draw.text(
        (240, legend_y + 28),
        "模型权重 weights/*.h5    前端数据 results/*.json    可视化页面 dashboard_fixed.html    论文与答辩材料 *.md",
        font=small_font,
        fill="#526071",
    )
    draw.text((116, legend_y + 68), "系统边界：", font=small_font, fill="#18202f")
    draw.text(
        (240, legend_y + 68),
        "本项目为交通风险预测与分析原型，重点展示离线训练、预测解释和可视化验证流程。",
        font=small_font,
        fill="#526071",
    )

    img.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
