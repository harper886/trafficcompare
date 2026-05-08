from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets"


def load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_center(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2 - 1), text, font=font, fill=fill)


def draw_table(title, columns, rows, output_name, col_widths=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    title_font = load_font(42, True)
    header_font = load_font(22, True)
    cell_font = load_font(21)
    cell_bold = load_font(21, True)
    note_font = load_font(18)

    left = 70
    top = 132
    row_h = 52
    title_h = 70
    bottom_pad = 60

    if col_widths is None:
        first = 150
        other = 135
        col_widths = [first] + [other] * (len(columns) - 1)

    table_w = sum(col_widths)
    width = left * 2 + table_w
    height = top + row_h * (len(rows) + 1) + bottom_pad

    img = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(img)

    navy = "#0f172a"
    muted = "#64748b"
    border = "#cbd5e1"
    header_bg = "#dbeafe"
    first_col_bg = "#eef2ff"
    myplan_bg = "#dcfce7"
    alt_bg = "#ffffff"
    alt_bg2 = "#f8fafc"
    accent = "#2563eb"
    green = "#15803d"

    draw.text((left, 44), title, font=title_font, fill=navy)
    draw.text((left, 94), "数值按原表逐项录入，保留四位小数。", font=note_font, fill=muted)

    x = left
    y = top

    for col_idx, (col, cw) in enumerate(zip(columns, col_widths)):
        fill = header_bg if col_idx != len(columns) - 1 else myplan_bg
        draw.rectangle((x, y, x + cw, y + row_h), fill=fill, outline=border, width=2)
        text_color = green if col == "Myplan" else navy
        draw_center(draw, (x, y, x + cw, y + row_h), col, header_font, text_color)
        x += cw

    y += row_h
    for row_idx, row in enumerate(rows):
        x = left
        for col_idx, (text, cw) in enumerate(zip(row, col_widths)):
            if col_idx == 0:
                fill = first_col_bg
                fnt = cell_bold
                color = accent
            elif col_idx == len(columns) - 1:
                fill = myplan_bg
                fnt = cell_bold
                color = green
            else:
                fill = alt_bg if row_idx % 2 == 0 else alt_bg2
                fnt = cell_font
                color = "#334155"
            draw.rectangle((x, y, x + cw, y + row_h), fill=fill, outline=border, width=2)
            draw_center(draw, (x, y, x + cw, y + row_h), text, fnt, color)
            x += cw
        y += row_h

    draw.rounded_rectangle(
        (left - 10, top - 10, left + table_w + 10, top + row_h * (len(rows) + 1) + 10),
        radius=18,
        outline="#94a3b8",
        width=2,
    )

    out = OUT_DIR / output_name
    img.save(out, "PNG", optimize=True)
    return out


def main():
    model_columns = [
        "方法", "GSNet", "STG2Seq", "ConvLSTM", "LSTM", "LightGBM", "CatBoost",
        "XGBoost", "LR", "ARIMA", "HA", "SNIPER", "Myplan"
    ]

    nyc_model_rows = [
        ["AUC-PR", "0.5890", "0.5083", "0.6078", "0.5899", "0.6341", "0.6134", "0.6201", "0.6102", "0.2394", "0.5827", "0.6240", "0.6955"],
        ["AUC-ROC", "0.8469", "0.7640", "0.8410", "0.7697", "0.8490", "0.8365", "0.8417", "0.8372", "0.5068", "0.8277", "0.8507", "0.8786"],
        ["F1 score", "0.6128", "0.5650", "0.5967", "0.6155", "0.6141", "0.6109", "0.6116", "0.6079", "0.2875", "0.6103", "0.6262", "0.6443"],
        ["Accuracy", "0.8291", "0.7959", "0.8057", "0.8068", "0.8204", "0.8097", "0.8143", "0.8081", "0.5954", "0.7944", "0.8366", "0.7865"],
        ["Recall", "0.7485", "0.6789", "0.7305", "0.7914", "0.7569", "0.7542", "0.7548", "0.7521", "0.2941", "0.7582", "0.7004", "0.8265"],
    ]

    chicago_model_rows = [
        ["AUC-PR", "0.4662", "0.3893", "0.4616", "0.4046", "0.4458", "0.4605", "0.4673", "0.4352", "0.1442", "0.4680", "0.4704", "0.5617"],
        ["AUC-ROC", "0.7809", "0.6657", "0.7450", "0.7676", "0.7894", "0.7792", "0.7804", "0.7786", "0.4994", "0.7540", "0.7829", "0.8290"],
        ["F1 score", "0.3872", "0.3710", "0.3974", "0.3782", "0.4059", "0.3971", "0.4023", "0.4004", "0.2240", "0.4064", "0.4137", "0.4730"],
        ["Accuracy", "0.8657", "0.8499", "0.8704", "0.8668", "0.8617", "0.8697", "0.8792", "0.8770", "0.7833", "0.8382", "0.8912", "0.7733"],
        ["Recall", "0.4351", "0.4135", "0.4485", "0.4223", "0.4608", "0.4480", "0.4545", "0.4523", "0.2357", "0.4621", "0.2694", "0.7055"],
    ]

    model_col_widths = [150, 118, 135, 145, 112, 145, 145, 135, 92, 112, 92, 125, 125]

    ablation_columns = [
        "方法", "Myplan（不加自适应和后处理）", "Myplan（不加自适应）", "Myplan（不加后处理）", "Myplan"
    ]

    nyc_ablation_rows = [
        ["AUC-PR", "0.6252", "0.6601", "0.6672", "0.6955"],
        ["AUC-ROC", "0.8486", "0.8642", "0.8652", "0.8786"],
        ["F1 score", "0.6087", "0.6228", "0.6348", "0.6443"],
        ["Accuracy", "0.7862", "0.7720", "0.8006", "0.7865"],
        ["Recall", "0.7108", "0.8046", "0.7407", "0.8265"],
    ]

    chicago_ablation_rows = [
        ["AUC-PR", "0.4725", "0.5176", "0.5136", "0.5617"],
        ["AUC-ROC", "0.7840", "0.8071", "0.8191", "0.8290"],
        ["F1 score", "0.4137", "0.4433", "0.4465", "0.4730"],
        ["Accuracy", "0.7417", "0.7576", "0.7889", "0.7733"],
        ["Recall", "0.6318", "0.6693", "0.6401", "0.7055"],
    ]

    ablation_col_widths = [150, 360, 280, 280, 145]

    outputs = [
        draw_table("模型比较结果表：NYC 数据集", model_columns, nyc_model_rows, "table_model_comparison_nyc.png", model_col_widths),
        draw_table("模型比较结果表：Chicago 数据集", model_columns, chicago_model_rows, "table_model_comparison_chicago.png", model_col_widths),
        draw_table("消融实验结果表：NYC 数据集", ablation_columns, nyc_ablation_rows, "table_ablation_nyc.png", ablation_col_widths),
        draw_table("消融实验结果表：Chicago 数据集", ablation_columns, chicago_ablation_rows, "table_ablation_chicago.png", ablation_col_widths),
    ]

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
