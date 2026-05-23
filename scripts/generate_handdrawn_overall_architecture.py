from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "model_overall_architecture_flow_handdrawn_large.png"

W, H = 3840, 2160
random.seed(42)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


F_TITLE = font(112, True)
F_SUB = font(48)
F_H1 = font(60, True)
F_H2 = font(45, True)
F_TEXT = font(42)
F_SMALL = font(36)
F_NOTE = font(41)


def rounded_polyline(draw: ImageDraw.ImageDraw, pts, fill, width=5, jitter=2):
    jpts = []
    for x, y in pts:
        jpts.append((x + random.uniform(-jitter, jitter), y + random.uniform(-jitter, jitter)))
    draw.line(jpts, fill=fill, width=width, joint="curve")


def hand_rect(draw: ImageDraw.ImageDraw, box, outline, fill, width=5, radius=26, passes=2):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    for _ in range(passes):
        j = 3
        jb = (
            x1 + random.uniform(-j, j),
            y1 + random.uniform(-j, j),
            x2 + random.uniform(-j, j),
            y2 + random.uniform(-j, j),
        )
        draw.rounded_rectangle(jb, radius=radius + random.uniform(-2, 2), outline=outline, width=width)


def arrow(draw, start, end, color=(54, 65, 79), width=7):
    x1, y1 = start
    x2, y2 = end
    midx = (x1 + x2) / 2
    pts = [(x1, y1), (midx, y1 + random.uniform(-6, 6)), (x2, y2)]
    rounded_polyline(draw, pts, color, width=width, jitter=1.5)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 22
    left = (x2 - size * math.cos(ang - 0.55), y2 - size * math.sin(ang - 0.55))
    right = (x2 - size * math.cos(ang + 0.55), y2 - size * math.sin(ang + 0.55))
    draw.polygon([(x2, y2), left, right], fill=color)


def wrap_text(draw, text, fnt, max_width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for ch in para:
            test = line + ch
            if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return lines


def center_text(draw, box, text, fnt, fill=(41, 50, 65), spacing=10):
    x1, y1, x2, y2 = box
    lines = wrap_text(draw, text, fnt, x2 - x1 - 36)
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2
    for line, h in zip(lines, heights):
        tw = draw.textbbox((0, 0), line, font=fnt)[2]
        draw.text((x1 + (x2 - x1 - tw) / 2, y), line, font=fnt, fill=fill)
        y += h + spacing


def label(draw, xy, text, color=(92, 105, 122)):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=F_SMALL)
    pad_x, pad_y = 16, 8
    rect = (x, y, x + bbox[2] + pad_x * 2, y + bbox[3] + pad_y * 2)
    draw.rounded_rectangle(rect, radius=12, fill=(250, 251, 252), outline=(214, 221, 230), width=2)
    draw.text((x + pad_x, y + pad_y - 2), text, font=F_SMALL, fill=color)


def module(draw, box, idx, title, body, outline, fill):
    hand_rect(draw, box, outline=outline, fill=fill, width=5, radius=30, passes=2)
    x1, y1, x2, y2 = box
    badge = (x1 + 32, y1 + 32, x1 + 92, y1 + 92)
    draw.rounded_rectangle(badge, radius=14, fill=(255, 255, 255), outline=outline, width=3)
    draw.text((x1 + 48, y1 + 34), str(idx), font=F_H2, fill=outline, anchor="la")
    draw.text((x1 + 116, y1 + 28), title, font=F_H1, fill=(18, 26, 39))
    center_text(draw, (x1 + 22, y1 + 124, x2 - 22, y2 - 20), body, F_TEXT)


def bullet_panel(draw, box, title, items, outline=(194, 203, 215)):
    hand_rect(draw, box, outline=outline, fill=(255, 255, 255), width=4, radius=24, passes=2)
    x1, y1, x2, y2 = box
    draw.text((x1 + 56, y1 + 46), title, font=F_H1, fill=(18, 26, 39))
    y = y1 + 140
    for key, val in items:
        draw.text((x1 + 58, y), key, font=F_TEXT, fill=(74, 114, 184))
        draw.text((x1 + 290, y), val, font=F_TEXT, fill=(58, 67, 82))
        y += 66


def main():
    img = Image.new("RGB", (W, H), (247, 249, 251))
    # Subtle paper texture.
    pix = img.load()
    for _ in range(120000):
        x = random.randrange(W)
        y = random.randrange(H)
        base = pix[x, y][0]
        delta = random.choice([-2, -1, 1, 2])
        v = max(238, min(255, base + delta))
        pix[x, y] = (v, v, min(255, v + 1))

    draw = ImageDraw.Draw(img)

    draw.text((150, 62), "模型总体架构流程图", font=F_TITLE, fill=(17, 24, 39))
    draw.text(
        (154, 202),
        "图 3-1  交通碰撞风险预测模型从输入构建、动态演化、空间注意力、时空融合到风险输出的整体流程",
        font=F_SUB,
        fill=(101, 116, 136),
    )

    muted = {
        "blue": ((91, 129, 173), (235, 241, 248)),
        "cyan": ((75, 154, 166), (233, 247, 248)),
        "green": ((83, 148, 100), (235, 247, 238)),
        "orange": ((180, 119, 62), (250, 242, 232)),
        "red": ((190, 103, 117), (251, 238, 240)),
    }

    boxes = [
        (140, 420, 730, 925),
        (865, 420, 1465, 925),
        (1590, 420, 2200, 925),
        (2315, 420, 2935, 925),
        (3040, 420, 3650, 925),
    ]
    titles = ["输入构建", "动态特征演化", "多源空间注意力", "时空融合建模", "风险预测输出"]
    bodies = [
        "交通时序特征 X\n事故标签 y\n阈值信号 threshold_nc\n区域映射 dict_xy\n形成滑动窗口：T × R × F",
        "Evolution 层\nUpdate / Decay 更新\n平滑门控制变化幅度\n输出动态状态：H_dynamic",
        "Road / POI / Record\n三类邻接关系\nScaled Dot Attention\n多轮空间传播\n输出空间增强特征",
        "动态特征 + 静态特征\n拼接后输入 ConvLSTM2D\n学习时间依赖与区域关联\n输出融合表示 Z",
        "Dense + Sigmoid\n得到每个区域风险概率\n阈值判定 + Streaming\n后处理输出预警结果",
    ]
    keys = ["blue", "cyan", "green", "orange", "red"]
    for i, box in enumerate(boxes):
        outline, fill = muted[keys[i]]
        module(draw, box, i + 1, titles[i], bodies[i], outline, fill)

    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(draw, (left[2] + 18, (left[1] + left[3]) // 2), (right[0] - 18, (right[1] + right[3]) // 2))

    label(draw, (780, 372), "滑动窗口序列")
    label(draw, (1510, 372), "动态状态 + 原始特征")
    label(draw, (2230, 372), "空间增强表示")
    label(draw, (2960, 372), "融合特征向量")

    bullet_panel(
        draw,
        (140, 1185, 1090, 1600),
        "模型训练与评估输出",
        [
            ("训练目标：", "Focal Loss + 动态差分约束"),
            ("评价指标：", "PR-AUC / ROC-AUC / F1 / Recall / Acc"),
            ("实验方式：", "基线对比 + 消融实验"),
        ],
    )

    relation_box = (1160, 1185, 2855, 1600)
    hand_rect(draw, relation_box, outline=(194, 203, 215), fill=(255, 255, 255), width=4, radius=24, passes=2)
    draw.text((1195, 1262), "空间关系与注意力融合细节", font=F_H1, fill=(18, 26, 39))
    smalls = [
        ((1215, 1385, 1665, 1532), "道路邻接\nroad_ad.txt", (73, 139, 160), (230, 245, 248)),
        ((1738, 1385, 2188, 1532), "功能邻接\npoi_ad.txt", (78, 145, 92), (232, 247, 235)),
        ((2260, 1385, 2710, 1532), "历史碰撞邻接\nrecord_ad.txt", (178, 120, 55), (251, 243, 224)),
    ]
    targets = [(1805, 910), (1895, 910), (2088, 910)]
    for (sb, txt, oc, fc), target in zip(smalls, targets):
        hand_rect(draw, sb, outline=oc, fill=fc, width=3, radius=18, passes=2)
        center_text(draw, sb, txt, F_TEXT)
        sx = (sb[0] + sb[2]) // 2
        arrow(draw, (sx, sb[1] - 10), target, color=oc, width=5)
    center_text(
        draw,
        (1190, 1532, 2800, 1590),
        "三类邻接分别计算注意力权重，再通过可学习权重融合为区域空间表示",
        F_SMALL,
        fill=(94, 109, 130),
    )

    out_panel = (2945, 1185, 3650, 1600)
    hand_rect(draw, out_panel, outline=(194, 203, 215), fill=(255, 255, 255), width=4, radius=24, passes=2)
    draw.text((3005, 1262), "前端展示输出", font=F_H1, fill=(18, 26, 39))
    y = 1350
    for item in ["风险热力图", "TP / FP / FN 标签校验", "区域邻接解释", "代表性时间帧 JSON"]:
        draw.ellipse((3010, y + 10, 3030, y + 30), fill=(176, 70, 91))
        draw.text((3052, y), item, font=F_TEXT, fill=(58, 67, 82))
        y += 55
    arrow(draw, (3320, 935), (3265, 1170), color=(176, 70, 91), width=6)
    label(draw, (2880, 1138), "部署展示阶段输出前端 JSON")

    note = (150, 1785, 3630, 1942)
    hand_rect(draw, note, outline=(214, 221, 230), fill=(255, 255, 255), width=3, radius=22, passes=1)
    draw.text((218, 1835), "图注：", font=F_H1, fill=(18, 26, 39))
    note_text = (
        "模型首先将交通特征、事故标签、阈值信号和区域邻接关系组织为时序窗口；随后通过动态演化层捕捉风险状态变化，"
        "再由多源空间注意力融合道路、POI 与历史碰撞邻接信息；最后经 ConvLSTM 完成时空联合建模，输出区域碰撞风险概率。"
    )
    lines = wrap_text(draw, note_text, F_NOTE, 3150)
    yy = 1830
    for line in lines:
        draw.text((360, yy), line, font=F_NOTE, fill=(65, 76, 94))
        yy += 48

    # Very light blur on background texture only is avoided; keep text sharp.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=95)


if __name__ == "__main__":
    main()
