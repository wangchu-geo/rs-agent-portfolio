# -*- coding: utf-8 -*-
"""
Portfolio 成果图件生成（P0）：产品代表性成果图 → portfolio/figures/
风格规范：浅色背景、单色系 sequential 渐变、中文标签、直接标注关键统计。
说明：水体总磷自产图效果不佳，展示统一用产品手册原图 doc_tp_example.png
（fig_tp 保留备用，__main__ 中默认不生成）。
运行：PYTHONUTF8=1 python portfolio/make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def read_valid(path, nodata_vals=(-9999,)):
    """读 tif，返回 (data, valid_mask)；valid 掩膜排除 nodata/NaN/0（可配置）。"""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
    valid = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.isclose(data, nodata)
    for v in nodata_vals:
        valid &= ~np.isclose(data, v)
    return data, valid


def finalize(fig, ax, path, title, note):
    ax.set_title(title, fontsize=15, pad=12)
    ax.text(0.01, -0.10, note, transform=ax.transAxes, fontsize=10, color="#555555")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已生成 {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# 1. 火点识别（Himawari-9，2025-04-15 03:00 UTC）
# ---------------------------------------------------------------------------
def fig_fire():
    d = r"E:/YYR/fire/work/NC_H09_20250415_0300"
    bg, bgv = read_valid(os.path.join(d, "H09_B07_cloud_masked_cleaned.tif"))
    fp, fpv = read_valid(os.path.join(d, "H09_fire_point.tif"))
    fp = (fp > 0) & fpv
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(bg, cmap="Greys", vmin=np.percentile(bg[bgv], 2),
              vmax=np.percentile(bg[bgv], 98))
    if fp.any():
        ys, xs = np.where(fp)
        ax.scatter(xs, ys, s=4, c="#d62728", label=f"火点像元 {fp.sum()} 个",
                   edgecolors="none")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_xticks([]), ax.set_yticks([])
    finalize(fig, ax, os.path.join(FIG_DIR, "fire_points_20250415.png"),
             "Himawari-9 森林火点识别（2025-04-15 03:00 UTC）",
             "底图：B07 红外亮温（去云后）；红点：上下文火点识别结果（验证 86 项断言全 PASS）")


# ---------------------------------------------------------------------------
# 2. 土壤相对湿度反演（S1+S2 双星协同，2026-05-13）
# ---------------------------------------------------------------------------
def fig_soil():
    d, v = read_valid(r"E:/YYR/turangshuifen/work/20260513/result/soil_moisture_map.tif")
    fig, ax = plt.subplots(figsize=(8, 6.2))
    im = ax.imshow(np.ma.masked_where(~v, d), cmap="YlGnBu",
                   vmin=float(d[v].min()), vmax=float(d[v].max()))
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="土壤相对湿度")
    cb.ax.yaxis.label.set_size(10)
    ax.set_xticks([]), ax.set_yticks([])
    # 标题过长会与图/色条堆叠：换行 + 收紧字号
    ax.set_title("土壤相对湿度反演\n（Sentinel-1 + Sentinel-2 双星协同，2026-05-13）",
                 fontsize=13, pad=14, linespacing=1.5)
    note = (f"随机森林反演，值域 {d[v].min():.2f}~{d[v].max():.2f}\n"
            "验证 33+24 项断言全 PASS（样本为模拟数据，评估口径=流程证明）")
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9,
             color="#555555", linespacing=1.5)
    fig.savefig(os.path.join(FIG_DIR, "soil_moisture_20260513.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成 soil_moisture_20260513.png")


# ---------------------------------------------------------------------------
# 3. PM10 反演（tianbu 观测校正后，2025-02-10 12:00 北京时间）
# ---------------------------------------------------------------------------
def fig_pm10():
    d, v = read_valid(
        r"E:/YYR/PM10(2.5)/work/20250210_12/correct/20250210_1200_校正结果_2KM_tianbu.tif")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.ma.masked_where(~v, d), cmap="magma",
                   vmin=float(d[v].min()), vmax=float(d[v].max()))
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"PM$_{10}$ ($\mu$g/m$^3$)")
    ax.set_xticks([]), ax.set_yticks([])
    finalize(fig, ax, os.path.join(FIG_DIR, "pm10_20250210.png"),
             "PM10 浓度反演（多源特征 RF + 站点观测校正，2025-02-10 12:00）",
             f"tianbu 校正后值域 {d[v].min():.1f}~{d[v].max():.1f} μg/m³；"
             "垂直廓线三维推算另见 sanwei 产物（验证 34 项断言全 PASS）")


# ---------------------------------------------------------------------------
# 4. 水体总磷反演（S2+S3 双传感器，2024-08-28）
# 注：自产图值域窄、观感差，展示统一用产品手册原图 doc_tp_example.png，
# 本函数保留备用（__main__ 默认不调用）。
# ---------------------------------------------------------------------------
def fig_tp():
    d, v = read_valid(r"E:/YYR/P_water/work/20240828_单景/result/pred_merged_correct_clip.tif")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.ma.masked_where(~v, d), cmap="YlOrRd",
                   vmin=float(d[v].min()), vmax=float(d[v].max()))
    fig.colorbar(im, ax=ax, shrink=0.8, label="总磷 (mg/L)")
    ax.set_xticks([]), ax.set_yticks([])
    finalize(fig, ax, os.path.join(FIG_DIR, "tp_water_20240828.png"),
             "水体总磷反演（Sentinel-2 + Sentinel-3，2024-08-28 单景）",
             f"反演值域 {d[v].min():.4f}~{d[v].max():.4f} mg/L（均优于 IV 类水界限 0.2 mg/L），"
             "已按水体矢量裁剪；验证 103 项断言全 PASS，R²=0.3157 仅机械对照")


# ---------------------------------------------------------------------------
# 5. NDVI 植被指数（基线产品）
# ---------------------------------------------------------------------------
def fig_ndvi():
    d, v = read_valid(r"E:/YYR/NDVI/result/NDVI_clip.tif")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.ma.masked_where(~v, d), cmap="YlGn", vmin=-0.2, vmax=0.9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
    ax.set_xticks([]), ax.set_yticks([])
    finalize(fig, ax, os.path.join(FIG_DIR, "ndvi_clip.png"),
             "NDVI 植被指数（Sentinel-2，Agent 工具链首个产品）",
             f"均值 {d[v].mean():.3f}，值域 {d[v].min():.3f}~{d[v].max():.3f}；"
             "支持多时相对比与中文解读报告")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_fire()
    fig_soil()
    fig_pm10()
    # fig_tp()  # 展示用产品手册原图 doc_tp_example.png，自产图按需取消注释
    fig_ndvi()
    print("全部图件生成完毕")
