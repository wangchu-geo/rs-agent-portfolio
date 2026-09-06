# -*- coding: utf-8 -*-
"""
水体总磷反演工具链

旧链：E:\\YYR\\P_water\\水体磷反演总结\\（8 个 py 脚本 + 3 个 SNAP graph XML，本模块
逐字复刻其语义并封装为 Agent 工具）。与其它产品不同：样本点只有 4 个监测站，
空间上难以大面积反演 → 拓宽时间线匹配样本（2024-07-01~12-31，GEE 提取 S2/S3 波段值
+ 日均实测总磷按 [date, point_name] 匹配）。

口径：
    - 站点总磷为真实逐小时监测数据（非随机模拟），但建模样本仅 45 个（3 站有效，
      清江桥总磷全 NaN 被剔除）→ 测试集 R²=0.3157 只做机械对照，评估=流程证明非精度；
    - 主验证 = 20240828 一景（S2A_MSIL2A_20240828T033531 +
      S3A_OL_1_EFR_20240828T032825 同一天成像）；
    - 训练特征来自 GEE 点提取（test/S2.csv、S3.csv），反演输入来自本地 SNAP 链 tif
      （300m→10m bilinear 重采样）——两链口径差异为结构性风险，README 如实注明；
    - 旧目录（水体磷反演总结/）只读；新产物写 E:/YYR/P_water/work/20240828_单景/。

旧脚本 ↔ 工具映射：
    csv_mean.py      →  pw_daily_mean            （逐小时监测→日均，date int 被平均成 float）
    csv合并.py       →  pw_merge_spatiotemporal  （S3+日均+S2 前缀化后 [date,point_name] inner join）
    手动 Excel 步骤  →  pw_build_samples         （去空值行+删多余字段→45 行建模样本集）
    特征计算.py      →  pw_compute_features      （6 波段组合指数，CSV+tif 双输出）
    磷反演-rf.py     →  pw_train_rf / pw_predict_rf（特征选择+RF 训练 / 全图反演）
    S2波段处理.py    →  pw_prepare_s2            （gpt 预处理→img→tif→重投影+水体裁剪）
    S3波段处理.py    →  pw_prepare_s3            （gpt 预处理→img→tif→对齐 S2 网格+NaN 掩膜）
    clip.py          →  pw_clip_product          （反演结果按水体矢量裁剪）

已实证旧品缺陷（correct 模式默认修复，legacy 模式逐字复现用于验证）：
    1. GEE 掩膜值未清洗入模：S3.csv 中 Oa01/Oa10 两列 820/832 为 -2.1e9 掩膜坏值，
       经合并/手动处理后 feature.csv 45 行中这两列仍 100% 坏值（非严格常数，6 个唯一值，
       实测 corr=0.0494 非 NaN）→ 旧品 threshold=0 时 38 特征全入模型（含坏值列）。
    2. 特征选择阈值 0 vs 文档 0.3（磷反演-rf.py:150 代码 vs 文档口径）：
       threshold=0 实际不筛选特征（|corr|>0 几乎全保留）。
    3. 特征计算 CSV÷0→inf 与 tif÷0→0 语义不一致（特征计算.py:52 df 直除 vs :116
       np.divide(where=data2!=0)）。
    4. 模型/scaler 未保存不可复现（磷反演-rf.py:162-165 训练后未落盘）。
    5. .enp 垃圾文件残留（旧 tif_2/Oa01_radiance.tif.enp、Oa03_radiance.tif.enp、
       result/pred_merged.tif.enp）。
    6. 训练特征=GEE 点提取、反演输入=SNAP 本地链重采样——两链口径差异
       （结构性风险，非代码 bug，README 如实注明）。

数据事实（实测）：
    - 监测 4 站逐时 CSV gbk（列：监测时间/date int/水质类别/10 参数）；监测/ 内有杂散
      副本"日均统计结果.csv"（utf-8-sig，gbk 读失败被旧脚本 try/except 跳过）；
    - 编码四分：监测/S2/S3/feature.csv=gbk；日均/s2-s3=utf-8-sig；data/feature.csv=utf-8；
    - nodata 四分：S2 clip 链=0.0（旧品 dstNodata='--nodata' 字符串实际继承源）、
      Oa 链=nan、指数 tif=0.0、反演产物=-9999；
    - 反演基准：pred_merged.tif 3184×1858 @EPSG:4326，有效像元 119721，值域
      0.07636~0.14077（总磷 mg/L 实测 0.0577~0.1812）；
    - legacy 训练参照值：best_params={max_depth:None, max_features:None,
      min_samples_leaf:1, min_samples_split:2, n_estimators:200}、
      测试 R²=0.3157、RMSE=0.0244（sklearn 1.3.2 与 1.6.1 一致）。
"""

import json
import os
import sys
import glob
import shutil

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ndvi_agent"))
from pipeline_tools import _json_safe  # noqa: E402

gdal.UseExceptions()

# 目标列（旧品手动处理保留原名，无对应 tif → 不参与反演堆叠）
TARGET_COL = "日均统计结果_总磷(mg/L)"

# 6 个波段组合指数（特征计算.py:146-182 固定配置）
FEATURE_OPS = [
    {"cols": ["S2_B5", "S2_B4"], "op": "add", "new_col": "B5+B4"},
    {"cols": ["S2_B5", "S2_B4"], "op": "subtract", "new_col": "B5-B4"},
    {"cols": ["S2_B3", "S2_B5"], "op": "divide", "new_col": "B3_divide_B5"},
    {"cols": ["S2_B4", "S2_B11"], "op": "divide", "new_col": "B4_divide_B11"},
    {"cols": ["S2_B6", "S2_B5"], "op": "divide", "new_col": "B6_divide_B5"},
    {"cols": ["S2_B3", "S2_B9"], "op": "divide", "new_col": "B3_divide_B9"},
]

# 总磷值域护栏（mg/L，实测 0.0577~0.1812）
_TP_HARD = (0.0, 1.0)
_TP_SOFT = (0.05, 0.2)

# legacy 训练参照值（本机 sklearn 1.6.1 实测，与旧机 1.3.2 一致）
_LEGACY_REF = {
    "best_params": {"max_depth": None, "max_features": None,
                    "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 200},
    "test_r2": 0.3157,
    "test_rmse": 0.0244,
    "n_features": 38,
}


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _out(payload, **extra):
    """把结果 dict 转成 JSON 字符串（Agent Tool 返回约定）。"""
    payload.update(extra)
    return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)


def _tif_attrs(path):
    """只读地读一个 tif 的关键属性（盘查/验证用）。"""
    with rasterio.open(path) as src:
        a = src.read(1)
        valid = None
        if src.nodata is not None and np.isfinite(src.nodata):
            valid = a != src.nodata
        else:
            valid = ~np.isnan(a)
        vals = a[valid] if valid.any() else a[~np.isnan(a)] if (~np.isnan(a)).any() else a
        return {
            "shape": list(src.shape), "dtype": src.dtypes[0], "crs": str(src.crs),
            "nodata": (None if src.nodata is None else float(src.nodata)),
            "bounds": [round(float(b), 4) for b in src.bounds],
            "min": round(float(np.min(vals)), 6), "max": round(float(np.max(vals)), 6),
            "valid_pct": round(100 * float(valid.mean()), 1),
        }


def _read_csv_fallback(path):
    """复刻 csv合并.py:70-80 / 磷反演-rf.py:12-22 的多编码回退读取。"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法用任何已知编码读取文件: {path}")


# ---------------------------------------------------------------------------
# 工具 1：数据盘查
# ---------------------------------------------------------------------------

def pw_inspect_data(base_dir, work_dir=None):
    """盘查旧链数据树与旧品基准属性（只读）。"""
    checks = {
        "monitor_4stations": [os.path.isfile(os.path.join(
            base_dir, "test", "监测", f"{s}.csv"))
            for s in ("三川", "八角", "双江桥", "清江桥")],
        "monitor_stray_copy": os.path.isfile(os.path.join(
            base_dir, "test", "监测", "日均统计结果.csv")),
        "s2_csv": os.path.isfile(os.path.join(base_dir, "test", "S2.csv")),
        "s3_csv": os.path.isfile(os.path.join(base_dir, "test", "S3.csv")),
        "daily_csv": os.path.isfile(os.path.join(base_dir, "test", "日均统计结果.csv")),
        "s2s3_csv": os.path.isfile(os.path.join(base_dir, "test", "s2-s3.csv")),
        "feature_csv": os.path.isfile(os.path.join(base_dir, "test", "feature.csv")),
        "data_feature_csv": os.path.isfile(os.path.join(base_dir, "test", "data", "feature.csv")),
        "s2_zip": os.path.isdir(os.path.join(base_dir, "test", "data", "S2", "data_0")),
        "s3_zip": os.path.isdir(os.path.join(base_dir, "test", "data", "S3", "data_0")),
        "s2_clip": os.path.isdir(os.path.join(base_dir, "test", "data", "S2", "clip_3")),
        "s3_tif": os.path.isdir(os.path.join(base_dir, "test", "data", "S3", "tif_2")),
        "feat_dir": os.path.isdir(os.path.join(base_dir, "test", "data", "feature")),
        "pred": os.path.isfile(os.path.join(base_dir, "test", "data", "result", "pred_merged.tif")),
        "shp": os.path.isfile(os.path.join(base_dir, "矢量", "水体.shp")),
    }
    details = {}
    feat_dir = os.path.join(base_dir, "test", "data", "feature")
    if checks["feat_dir"]:
        details["feature_tif_count"] = len(
            [f for f in os.listdir(feat_dir) if f.endswith(".tif")])
    if checks["pred"]:
        details["pred_attrs"] = _tif_attrs(os.path.join(
            base_dir, "test", "data", "result", "pred_merged.tif"))
    if checks["feature_csv"]:
        try:
            df = pd.read_csv(os.path.join(base_dir, "test", "feature.csv"), encoding="gbk")
            details["feature_csv"] = {"shape": list(df.shape),
                                      "target_min": round(float(df[TARGET_COL].min()), 6),
                                      "target_max": round(float(df[TARGET_COL].max()), 6)}
        except Exception as e:  # noqa: BLE001
            details["feature_csv"] = {"error": str(e)}
    return _out({"status": "ok", "checks": checks, "details": details})


# ---------------------------------------------------------------------------
# 工具 2：逐小时监测 → 日均（csv_mean.py）
# ---------------------------------------------------------------------------

def pw_daily_mean(monitor_dir, out_csv, mode="correct"):
    """复刻 csv_mean.py:142-192 的逐站点日均计算。

    legacy：gbk 读取 + try/except 跳过读失败文件（复刻杂散副本跳过行为）、
            groupby(['point_name','day']).mean(numeric_only=True) 把 date int 列
            平均成 float（20240701.0，旧品巧合 bug 保留）。
    correct：同样输出但 date 列显式取日期整数（不靠 mean 巧合）。
    """
    if os.path.exists(out_csv):
        return _out({"status": "exists", "out_csv": out_csv})
    all_daily = []
    for filename in os.listdir(monitor_dir):
        if not filename.endswith(".csv"):
            continue
        site = os.path.splitext(filename)[0]
        path = os.path.join(monitor_dir, filename)
        try:
            df = pd.read_csv(path, encoding="gbk")
        except Exception:  # noqa: BLE001  —— 复刻旧品 try/except 跳过
            continue
        try:
            df["监测时间"] = pd.to_datetime(df["监测时间"])
            df["day"] = df["监测时间"].dt.date
            df["point_name"] = site
            daily = df.groupby(["point_name", "day"]).mean(numeric_only=True).reset_index()
            if mode == "correct" and "date" in daily.columns:
                daily["date"] = pd.to_datetime(daily["day"]).dt.strftime("%Y%m%d").astype(int)
            all_daily.append(daily)
        except Exception:  # noqa: BLE001
            continue
    if not all_daily:
        return _out({"status": "error", "message": "未找到任何有效CSV文件"})
    final = pd.concat(all_daily, ignore_index=True).sort_values(["day", "point_name"])
    final.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return _out({"status": "ok", "out_csv": out_csv,
                 "shape": list(final.shape), "columns": list(final.columns)})


# ---------------------------------------------------------------------------
# 工具 3：三 CSV 时空匹配（csv合并.py）
# ---------------------------------------------------------------------------

def pw_merge_spatiotemporal(s2_csv, s3_csv, daily_csv, out_csv, mode="correct"):
    """复刻 csv合并.py:70-117：三个 CSV 前缀化后按 [date, point_name] inner join。

    legacy：fallback 编码序（utf-8→utf-8-sig→gbk→gb2312→latin1）、前缀规则
            （date/point_name 不加前缀）、reduce 序 [S3, 日均, S2]、
            float/int date 数值匹配巧合保留。
    correct：date 统一 int64 后 merge（显式化巧合）。
    """
    if os.path.exists(out_csv):
        return _out({"status": "exists", "out_csv": out_csv})
    names = ("S3", os.path.splitext(os.path.basename(daily_csv))[0],
             os.path.splitext(os.path.basename(s2_csv))[0])
    processed = []
    for path, name in zip((s3_csv, daily_csv, s2_csv), names):
        df = _read_csv_fallback(path)
        if mode == "correct" and "date" in df.columns:
            df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")
        new_columns = []
        for col in df.columns:
            if col in ("date", "point_name"):
                new_columns.append(col)
            else:
                new_columns.append(f"{name}_{col}")
        df.columns = new_columns
        processed.append(df)
    merged = processed[0]
    for right in processed[1:]:
        merged = pd.merge(merged, right, on=["date", "point_name"], how="inner")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return _out({"status": "ok", "out_csv": out_csv,
                 "shape": list(merged.shape)})


# ---------------------------------------------------------------------------
# 工具 4：建模样本集构建（旧链手动 Excel 步骤工具化）
# ---------------------------------------------------------------------------

def pw_build_samples(merged_csv, out_csv, mode="correct"):
    """把 s2-s3.csv 按旧链"手动处理"规则转成建模样本集。

    33 列规则（按 s2-s3.csv 原列序）：
        1. 取 TARGET_COL（总磷，保留原名——无对应 tif，仅作目标）；
        2. 取所有 S3_OaXX_radiance 列（跳过 S3_vOa19_radiance），重命名去掉 S3_ 前缀
           （Oa 链 tif 文件名即 OaXX_radiance.tif，stem 需与列名一致）；
        3. 取所有 S2_B* 列（12 个），保留 S2_ 前缀（clip 产物名为 S2_B1.tif）。
    行规则：对上述列 dropna(how='any') → 45 行（行序保持 s2-s3 原序）。
    legacy 输出 gbk（复刻 Excel 保存编码）。

    correct：Oa 列 < -1e9 的掩膜坏值先转 NaN → 删全 NaN 列（Oa01/Oa10 100% 坏值
             被整列删除）→ 31 列 → 同样的 45 行。
    """
    if os.path.exists(out_csv):
        return _out({"status": "exists", "out_csv": out_csv})
    df = pd.read_csv(merged_csv, encoding="utf-8-sig")

    oa_cols = [c for c in df.columns
               if c.startswith("S3_Oa") and c.endswith("_radiance")
               and c != "S3_vOa19_radiance"]
    s2_cols = [c for c in df.columns if c.startswith("S2_B")]
    keep = [TARGET_COL] + oa_cols + s2_cols
    sub = df[keep].copy()
    rename = {c: c.replace("S3_", "", 1) for c in oa_cols}
    sub.rename(columns=rename, inplace=True)

    if mode == "correct":
        for c in sub.columns:
            if c.startswith("Oa") and sub[c].dtype != object:
                sub[c] = sub[c].where(sub[c] >= -1e9, np.nan)
        sub = sub.dropna(axis=1, how="all")

    n_before = len(sub)
    sub = sub.dropna(how="any")
    if mode == "legacy":
        # 复刻 Excel 手动保存的 dtype/精度行为：整数值单元格写回整数（旧品 20 个
        # Oa 列 int64、总磷+12 个 S2_B 列 float64）；总磷日均值在 Excel 中按
        # 9 位小数截断（30 个循环小数值如 0.1018333333333333→0.101833333，
        # 其余 15 个小数位≤9 不受影响；S2_B 列无损）
        for c in sub.columns:
            if np.issubdtype(sub[c].dtype, np.number) and (sub[c] % 1 == 0).all():
                sub[c] = sub[c].astype("int64")
            elif np.issubdtype(sub[c].dtype, np.number):
                sub[c] = sub[c].round(9)
    sub.to_csv(out_csv, index=False, encoding="gbk")
    return _out({"status": "ok", "out_csv": out_csv,
                 "shape": list(sub.shape),
                 "rows_before_dropna": n_before,
                 "columns": list(sub.columns)})


# ---------------------------------------------------------------------------
# 工具 5：S2 预处理链（S2波段处理.py）
# ---------------------------------------------------------------------------

def pw_prepare_s2(data_dir, xml_path, yuchuli_dir, tif_dir, clip_dir, shp_path,
                  mode="correct"):
    """S2 链：gpt 预处理（exists 复验不重跑）→ img→tif → 重投影+水体裁剪。

    legacy：gpt 段只做存在性复验（yuchuli .dim + .data/*.img 齐即通过，soil 先例）；
            img_to_tif 复刻（tif 写入 tif_dir/<同名子目录>/B*.tif，rasterio 逐波段拷贝
            不带 nodata → tif nodata=None）；
            clip 复刻 gdal.Warp(dstSRS=shp CRS + cutline cropToCutline + S2_ 前缀 +
            recursive 递归 glob)，dstNodata=0.0（旧品传 '--nodata' 字符串，GDAL
            atof() 解析失败返回 0.0 —— 实测 clip_3 nodata=0.0）。
    correct：dstNodata 显式 0.0（不再依赖 atof 字符串巧合）。
    """
    # exists 幂等短路：clip 12 个 + tif 12 个齐即视为已完成
    if len(glob.glob(os.path.join(clip_dir, "S2_*.tif"))) >= 12 and \
            len(glob.glob(os.path.join(tif_dir, "**", "*.tif"), recursive=True)) >= 12:
        return _out({"status": "exists", "clip_dir": clip_dir,
                     "clipped": len(glob.glob(os.path.join(clip_dir, "S2_*.tif")))})
    zips = [f for f in os.listdir(data_dir) if f.lower().endswith(".zip")]
    if not zips:
        return _out({"status": "error", "message": f"data_dir 无影像压缩包: {data_dir}"})
    base = os.path.splitext(zips[0])[0]
    dim = os.path.join(yuchuli_dir, base + ".dim")
    data_sub = os.path.join(yuchuli_dir, base + ".data")
    imgs = glob.glob(os.path.join(data_sub, "*.img"))
    gpt_ok = os.path.exists(dim) and os.path.isdir(data_sub) and len(imgs) >= 12
    if not gpt_ok:
        return _out({"status": "error",
                     "message": f"gpt 预处理产物缺失（需要先跑 SNAP graph: {xml_path}），"
                                f"dim={os.path.exists(dim)} data={os.path.isdir(data_sub)} "
                                f"img={len(imgs)}"})

    # img → tif（复刻 img_to_tif：遍历 yuchuli 下 *.SAFE.data 目录，
    # tif 写入 tif_dir/<同名子目录>/B*.tif；不带 nodata → tif nodata=None）
    os.makedirs(tif_dir, exist_ok=True)
    made = 0
    for entry in os.listdir(yuchuli_dir):
        if not entry.endswith("SAFE.data"):
            continue
        tif_sub = os.path.join(tif_dir, entry)
        os.makedirs(tif_sub, exist_ok=True)
        for img_file in glob.glob(os.path.join(yuchuli_dir, entry, "*.img")):
            tif_file = os.path.join(tif_sub, os.path.basename(img_file).replace(".img", ".tif"))
            with rasterio.open(img_file) as src:
                with rasterio.open(tif_file, "w", driver="GTiff", count=src.count,
                                   dtype=src.dtypes[0], crs=src.crs,
                                   transform=src.transform, width=src.width,
                                   height=src.height) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
            made += 1

    # 裁剪（复刻 batch_reproject_and_clip：dstSRS=shp CRS + cutline + S2_ 前缀 +
    # recursive glob；dstNodata=0.0 —— 旧品 '--nodata' 经 GDAL atof() 得 0.0）
    os.makedirs(clip_dir, exist_ok=True)
    shp_ds = gdal.OpenEx(shp_path)
    layer = shp_ds.GetLayer()
    target_wkt = layer.GetSpatialRef().ExportToWkt()
    shp_ds = None
    warped = 0
    for tif_path in glob.glob(os.path.join(tif_dir, "**", "*.tif"), recursive=True):
        if os.path.basename(tif_path).startswith("S2_"):
            continue
        out_path = os.path.join(clip_dir, "S2_" + os.path.basename(tif_path))
        gdal.Warp(out_path, tif_path,
                  format="GTiff", dstSRS=target_wkt,
                  cutlineDSName=shp_path, cropToCutline=True,
                  dstNodata=0.0,
                  creationOptions=["COMPRESS=LZW", "TILED=YES"])
        warped += 1
    return _out({"status": "ok", "gpt_verified": True, "img_to_tif": made,
                 "clipped": warped, "clip_dir": clip_dir})


# ---------------------------------------------------------------------------
# 工具 6：S3 预处理链（S3波段处理.py）
# ---------------------------------------------------------------------------

def pw_prepare_s3(data_dir, xml_path, yuchuli_dir, tif_dir, ref_tif, s2_clip_dir,
                  feature_dir, mode="correct"):
    """S3 链：gpt 预处理（exists 复验）→ img→tif → 对齐 S2 网格 + NaN 掩膜。

    legacy：img_to_tif 只转 *radiance.img（tif 写入 tif_dir/<同名子目录>/）；
            process_tifs 复刻——以 ref_tif 的 outputBounds+width/height 对齐、
            bilinear、参考 nodata=0 区置 NaN、SetNoDataValue(np.nan)；
            随后把 S2 clip 12 tif 拷入 feature_dir（旧链 feature 目录=两链组合）。
    correct：同语义，保证输出无 .enp 残留（旧品 feature 目录有 2 个 .enp 垃圾文件）。
    """
    # exists 幂等短路：feature_dir 已有 21 Oa + 12 S2_B 即视为已完成
    oa_in_feat = len(glob.glob(os.path.join(feature_dir, "Oa*_radiance.tif")))
    s2_in_feat = len(glob.glob(os.path.join(feature_dir, "S2_B*.tif")))
    if oa_in_feat >= 21 and s2_in_feat >= 12:
        return _out({"status": "exists", "feature_dir": feature_dir,
                     "oa_tifs": oa_in_feat, "s2_tifs": s2_in_feat})
    zips = [f for f in os.listdir(data_dir) if f.lower().endswith(".zip")]
    if not zips:
        return _out({"status": "error", "message": f"data_dir 无影像压缩包: {data_dir}"})
    base = os.path.splitext(zips[0])[0]
    dim = os.path.join(yuchuli_dir, base + ".dim")
    data_sub = os.path.join(yuchuli_dir, base + ".data")
    rad_imgs = glob.glob(os.path.join(data_sub, "*radiance.img"))
    gpt_ok = os.path.exists(dim) and os.path.isdir(data_sub) and len(rad_imgs) >= 21
    if not gpt_ok:
        return _out({"status": "error",
                     "message": f"gpt 预处理产物缺失（需要先跑 SNAP graph: {xml_path}），"
                                f"dim={os.path.exists(dim)} data={os.path.isdir(data_sub)} "
                                f"radiance_img={len(rad_imgs)}"})

    # img → tif（复刻 S3波段处理.py:185-212：*.data 目录 + *radiance.img，
    # tif 写入 tif_dir/<同名子目录>/）
    os.makedirs(tif_dir, exist_ok=True)
    made = 0
    data_subs = []
    for entry in os.listdir(yuchuli_dir):
        if not entry.endswith(".data"):
            continue
        data_subs.append(entry)
        tif_sub = os.path.join(tif_dir, entry)
        os.makedirs(tif_sub, exist_ok=True)
        for img_file in glob.glob(os.path.join(yuchuli_dir, entry, "*radiance.img")):
            tif_file = os.path.join(tif_sub, os.path.basename(img_file).replace(".img", ".tif"))
            with rasterio.open(img_file) as src:
                with rasterio.open(tif_file, "w", driver="GTiff", count=src.count,
                                   dtype=src.dtypes[0], crs=src.crs,
                                   transform=src.transform, width=src.width,
                                   height=src.height) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
            made += 1

    # process_tifs（复刻 S3波段处理.py:214-319：对齐参考网格 + NaN 掩膜）
    # 旧品 input_dir 是显式的嵌套子目录（tif_dir/<*.data>/）
    if len(data_subs) != 1:
        return _out({"status": "error",
                     "message": f"tif 子目录应为 1 个，实际 {len(data_subs)}"})
    input_dir = os.path.join(tif_dir, data_subs[0])
    os.makedirs(feature_dir, exist_ok=True)
    ref_ds = gdal.Open(ref_tif)
    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
    xmin = ref_gt[0]
    ymax = ref_gt[3]
    xmax = xmin + ref_gt[1] * cols
    ymin = ymax + ref_gt[5] * rows
    ref_band = ref_ds.GetRasterBand(1)
    ref_data = ref_band.ReadAsArray()
    ref_nodata = ref_band.GetNoDataValue()
    if ref_nodata is not None:
        valid_mask = ~np.isclose(ref_data, ref_nodata, equal_nan=True)
    else:
        valid_mask = ~np.isnan(ref_data)
    valid_mask = valid_mask.astype(bool)
    ref_ds = None

    processed = 0
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".tif", ".tiff")):
            continue
        input_path = os.path.join(input_dir, filename)
        temp_path = os.path.join(feature_dir, f"temp_{filename}")
        final_path = os.path.join(feature_dir, filename)
        ds = gdal.Warp(temp_path, input_path,
                       outputBounds=(xmin, ymin, xmax, ymax),
                       width=cols, height=rows, dstSRS=ref_proj,
                       format="GTiff", resampleAlg=gdal.GRA_Bilinear,
                       outputType=gdal.GDT_Float32,
                       srcNodata=None, dstNodata=None)
        if not ds:
            continue
        ds = None
        temp_ds = gdal.Open(temp_path, gdal.GA_Update)
        if temp_ds is None:
            continue
        for b in range(1, temp_ds.RasterCount + 1):
            band = temp_ds.GetRasterBand(b)
            data = band.ReadAsArray().astype(np.float32)
            data[~valid_mask] = np.nan
            band.WriteArray(data)
            try:
                band.SetNoDataValue(np.nan)
            except Exception:  # noqa: BLE001
                pass
            band.FlushCache()
        temp_ds = None
        os.replace(temp_path, final_path)
        processed += 1

    # S2 clip 12 tif 拷入 feature_dir（旧链手动步骤：所有特征放同一文件夹）
    s2_copied = 0
    for f in os.listdir(s2_clip_dir):
        if f.startswith("S2_") and f.endswith(".tif"):
            shutil.copy2(os.path.join(s2_clip_dir, f), os.path.join(feature_dir, f))
            s2_copied += 1
    return _out({"status": "ok", "gpt_verified": True, "img_to_tif": made,
                 "aligned": processed, "s2_copied": s2_copied,
                 "feature_dir": feature_dir})


# ---------------------------------------------------------------------------
# 工具 7：特征计算（特征计算.py）
# ---------------------------------------------------------------------------

def pw_compute_features(csv_path, tif_dir, out_csv, out_tif_dir, mode="correct"):
    """6 个波段组合指数：CSV 加列 + tif 生成（特征计算.py:8-135 复刻）。

    legacy：CSV gbk 读、df 直除（÷0→inf 语义保留；本集实测无 inf）、
            CSV 输出 utf-8（默认 to_csv）；tif np.divide(where=data2!=0)（÷0 处得 0）、
            NaN→output_nodata(=第一个输入的 nodata，S2 链=0.0)、profile 继承第一输入。
    correct：CSV 与 tif 的 ÷0 语义统一为 0.0。
    """
    expected = [f"{op['new_col']}.tif" for op in FEATURE_OPS]
    existing = [f for f in expected if os.path.exists(os.path.join(out_tif_dir, f))]
    if len(existing) == len(expected) and os.path.exists(out_csv):
        return _out({"status": "exists", "out_csv": out_csv, "tifs": len(existing)})

    df = pd.read_csv(csv_path, encoding="gbk")
    for op in FEATURE_OPS:
        col1, col2 = op["cols"]
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"列 {col1}/{col2} 不存在于CSV文件中")
        if op["op"] == "add":
            df[op["new_col"]] = df[col1] + df[col2]
        elif op["op"] == "subtract":
            df[op["new_col"]] = df[col1] - df[col2]
        elif op["op"] == "multiply":
            df[op["new_col"]] = df[col1] * df[col2]
        elif op["op"] == "divide":
            if mode == "legacy":
                df[op["new_col"]] = df[col1] / df[col2]  # 旧品直除（÷0→inf）
            else:
                denom = df[col2].replace(0, np.nan)
                df[op["new_col"]] = df[col1] / denom
                df[op["new_col"]] = df[op["new_col"]].fillna(0.0)  # 与 tif where 语义一致
    df.to_csv(out_csv, index=False)
    os.makedirs(out_tif_dir, exist_ok=True)

    made = 0
    for op in FEATURE_OPS:
        col1, col2 = op["cols"]
        tif1 = os.path.join(tif_dir, f"{col1}.tif")
        tif2 = os.path.join(tif_dir, f"{col2}.tif")
        if not os.path.exists(tif1) or not os.path.exists(tif2):
            raise FileNotFoundError(f"找不到栅格文件: {tif1} 或 {tif2}")
        with rasterio.open(tif1) as src1:
            data1 = src1.read(1).astype(np.float32)
            profile = src1.profile.copy()
            nodata1 = src1.nodata
        with rasterio.open(tif2) as src2:
            data2 = src2.read(1).astype(np.float32)
            nodata2 = src2.nodata
            if src1.shape != src2.shape or src1.transform != src2.transform \
                    or src1.crs != src2.crs:
                raise ValueError(f"栅格参数不匹配: {col1} vs {col2}")
        if nodata1 is not None:
            data1[data1 == nodata1] = np.nan
        if nodata2 is not None:
            data2[data2 == nodata2] = np.nan
        if op["op"] == "add":
            result = data1 + data2
        elif op["op"] == "subtract":
            result = data1 - data2
        elif op["op"] == "multiply":
            result = data1 * data2
        else:
            result = np.divide(data1, data2, where=data2 != 0)
        output_nodata = nodata1 if nodata1 is not None else -9999.0
        result = np.nan_to_num(result, nan=output_nodata)
        profile.update(dtype=rasterio.float32, nodata=output_nodata, count=1,
                       driver="GTiff")
        out_path = os.path.join(out_tif_dir, f"{op['new_col']}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result.astype(rasterio.float32), 1)
        made += 1
    return _out({"status": "ok", "out_csv": out_csv, "tifs": made,
                 "columns": list(df.columns)})


# ---------------------------------------------------------------------------
# 工具 8：特征选择 + RF 训练（磷反演-rf.py:24-69）
# ---------------------------------------------------------------------------

def pw_train_rf(feature_csv, out_dir, mode="correct", target_col=TARGET_COL,
                correlation_threshold=0.3, random_state=42):
    """特征选择（|corr|>threshold）+ StandardScaler + GridSearchCV 训练。

    legacy：threshold=0 → 38 特征全入选（含坏值列 Oa01/Oa10，缺陷#1 复刻）；
    correct：默认 0.3（文档口径）。
    落盘 model.pkl / scaler.pkl / features.json / metrics.json
    （legacy 也落盘供反演工具用；旧品不落盘，README 注明仅作中间件）。
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    model_pkl = os.path.join(out_dir, "model.pkl")
    metrics_json = os.path.join(out_dir, "metrics.json")
    if os.path.exists(model_pkl) and os.path.exists(metrics_json):
        with open(metrics_json, encoding="utf-8") as f:
            m = json.load(f)
        return _out({"status": "exists", "out_dir": out_dir, **m})

    data = _read_csv_fallback(feature_csv)
    target = data[target_col]
    features = data.drop(target_col, axis=1)
    scaler_all = StandardScaler()
    scaled = pd.DataFrame(scaler_all.fit_transform(features), columns=features.columns)
    corr = scaled.corrwith(target).abs()
    if mode == "legacy":
        threshold = 0
    else:
        threshold = correlation_threshold
    selected = corr[corr > threshold].index.tolist()

    X = data[selected]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestRegressor(random_state=random_state)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train_scaled, y_train)
    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_test_scaled)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    importance = dict(zip(selected, best_rf.feature_importances_.astype(float).tolist()))
    metrics = {
        "n_input_features": int(features.shape[1]),
        "n_selected_features": len(selected),
        "selected_features": selected,
        "threshold": threshold,
        "best_params": grid.best_params_,
        "test_r2": round(r2, 6),
        "test_rmse": round(rmse, 6),
        "feature_importance": importance,
    }
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_rf, model_pkl)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    with open(os.path.join(out_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected}, f, ensure_ascii=False, indent=1)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    return _out({"status": "ok", "out_dir": out_dir, **metrics})


# ---------------------------------------------------------------------------
# 工具 9：全图反演（磷反演-rf.py:73-142）
# ---------------------------------------------------------------------------

def pw_predict_rf(feature_tif_dir, model_dir, out_tif, mode="correct"):
    """按选定特征堆叠 tif → scaler.transform → 预测 → NaN→-9999 写图。

    legacy：os.listdir 收集 tif、按 stem 匹配特征名、np.zeros 堆叠、
            valid_mask=np.all(np.isfinite)、meta 取第一个匹配 tif、
            输出 nodata=-9999（复刻 model_inversion 逐字语义）。
    correct：从 model_dir 加载 model/scaler/features（修复缺陷#4 后语义同一）。
    """
    if os.path.exists(out_tif):
        return _out({"status": "exists", "out_tif": out_tif})
    import joblib
    with open(os.path.join(model_dir, "features.json"), encoding="utf-8") as f:
        selected_features = json.load(f)["selected_features"]
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    raster_files = [f for f in os.listdir(feature_tif_dir) if f.endswith(".tif")]
    selected_files = [f for f in raster_files
                      if os.path.splitext(f)[0] in selected_features]
    missing = [feat for feat in selected_features
               if feat not in [os.path.splitext(f)[0] for f in selected_files]]
    if missing:
        return _out({"status": "error", "missing_tifs": missing})

    sample_path = os.path.join(feature_tif_dir, selected_files[0])
    with rasterio.open(sample_path) as src_sample:
        meta = src_sample.meta
        rows, cols = src_sample.height, src_sample.width
        meta.update(dtype="float32", nodata=-9999, count=1)

    feature_count = len(selected_features)
    stack = np.zeros((feature_count, rows, cols), dtype=np.float32)
    for idx, feat in enumerate(selected_features):
        file_path = os.path.join(feature_tif_dir, feat + ".tif")
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            if data.shape != (rows, cols):
                raise ValueError(f"栅格 {feat}.tif 形状 {data.shape} 与基准 ({rows},{cols}) 不一致")
            stack[idx, :, :] = data

    valid_mask = np.all(np.isfinite(stack), axis=0)
    output_array = np.full((rows, cols), np.nan, dtype=np.float32)
    if np.sum(valid_mask) > 0:
        flat_stack = stack.reshape(feature_count, -1)
        X_valid = flat_stack[:, valid_mask.ravel()].T
        X_valid_scaled = scaler.transform(X_valid)
        output_array[valid_mask] = model.predict(X_valid_scaled)
    output_array[np.isnan(output_array)] = -9999
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(output_array, 1)
    return _out({"status": "ok", "out_tif": out_tif,
                 "n_features": feature_count, "shape": [rows, cols],
                 "valid_pixels": int(np.sum(valid_mask))})


# ---------------------------------------------------------------------------
# 工具 10：反演结果水体裁剪（clip.py）
# ---------------------------------------------------------------------------

def pw_clip_product(pred_tif, shp_path, out_tif):
    """按水体.shp 裁剪反演结果（clip.py:10-47 语义；旧链未实际执行无对照物）。"""
    if os.path.exists(out_tif):
        return _out({"status": "exists", "out_tif": out_tif})
    gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
    gdal.Warp(out_tif, pred_tif,
              cutlineDSName=shp_path, cropToCutline=True,
              dstNodata=-9999, multithread=True,
              resampleAlg=gdal.GRA_NearestNeighbour)
    return _out({"status": "ok", "out_tif": out_tif, "attrs": _tif_attrs(out_tif)})
