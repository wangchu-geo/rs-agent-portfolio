# -*- coding: utf-8 -*-
"""土壤相对湿度反演工具链。

旧脚本与数据：E:\\YYR\\turangshuifen\\project\\（1.py=S1 链、S2.py=S2 链、RF_test.py=随机森林反演）。
本模块把它们工具化，供大语言模型通过 Tool Use 编排"数据盘查→S1/S2 特征→特征装配→训练→反演→评估"。

函数映射（新工具 → 旧函数，附旧脚本行号）：
    soil_inspect_data      —— 新增（阶段状态盘查）
    soil_s1_gpt            —— 1.py main() 379-394 行（gpt 哨兵1预处理.xml）
    soil_s1_features       —— 1.py img_to_tif(156-183)/calculate_sin_cos(185-234)/crop_rasters(278-321)
    soil_s2_gpt            —— S2.py main() 514-522 行（gpt S2chuli.xml）
    soil_s2_features       —— S2.py img_to_tif(173-200)/find_tif_files_by_date(214-241)/
                              process_date_groups(281-323)/reproject_and_crop(399-463)/process_tif_files(465-485)
    soil_assemble_features —— RF_test.py copy_tif_files_with_validation(80-91)/load_data_and_extract_features(96-171)
    soil_train_model       —— RF_test.py select_features(174-185)/train_and_validate_model(188-268)
    soil_predict_map       —— RF_test.py predict_soil_moisture(272-341)
    soil_assess_model      —— 新增（模型质量自动评估）

旧品两处已实证 bug（本模块默认修复，legacy 模式可复现）：
    [bug1 特征列错位] 训练特征按 sorted() 排序采样（RF_test.py 121 行），预测按 os.listdir() 顺序
        堆叠（RF_test.py 273 行）。实测 NTFS 目录枚举顺序 ≠ Python sorted（大小写不敏感排序，
        cos_rushejiao 排最前）→ 旧品预测时 20 个特征列与训练时错位。
    [bug2 无地理重投影] 预测时形状不符的层用 src.read(1, out_shape=..., resampling=bilinear)
        纯数组拉伸（RF_test.py 289-292 行），S2 20m 层被直接拉伸到 S1 10m 网格，没有
        dst_transform 地理对齐（实测两裁剪范围重合，偏移仅在子像元级，危害远小于 bug1）。
    另：np.resize 兜底（RF_test.py 296 行）实操无副作用（out_shape 已对齐形状），照写并注释。

不复刻的旧代码缺陷（与火点链同一"正确性优先"口径，均在 docstring 注明）：
    - 1.py 422-427 行 clip_file 未定义 NameError → 旧 main 的裁剪循环从未跑通，仅复刻 crop_rasters 逻辑
    - calculate_sin_cos 的正则 .*rushejiao\\.tif$ 会把 sin_/cos_ 输出再次当输入（重跑即污染），
      改为精确匹配 rushejiao.tif + 产物存在短路
    - S2.py 的 S2* 正则（S 后跟零或多个 2）无过滤能力，工具显式收 zip 路径，不用它
    - S2.py reproject_and_crop 的 rasterio.open("mem", "w+") 在 Windows 实际落盘为真实
      GTiff 垃圾文件（117MB），工具改 rasterio.io.MemoryFile 纯内存

产物约定：旧 test/sujiatun/ 只读；新产物写 work/<日期>/{S1,S2,feature,result}（镜像旧树）。
样本口径：soim_values 样本 CSV 为随机模拟数据（样本点与样本值均随机生成，无真实实测数据），
    本链目标=跑通反演流程证明反演能力；CV R²≈0 为随机目标预期，不做精度声明。
环境：Python 3.9（sklearn 1.6.1 / numpy 2.0.2 / rasterio 1.4.3）；Windows 需 PYTHONUTF8=1。
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ndvi_agent"))
from pipeline_tools import GPT_EXE, _json_safe  # noqa: E402

# ============================================================
# 常量
# ============================================================
DEFAULT_BASE_DIR = r"E:/YYR/turangshuifen"
DEFAULT_AREA = "sujiatun"
GRAPH_S1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs", "哨兵1预处理.xml")
GRAPH_S2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs", "S2chuli.xml")
GPT_TIMEOUT = 3600  # 单景 gpt 超时（秒）

# S1 链 9 特征（哨兵1预处理.xml 导出 7 波段 + sin/cos 入射角）
S1_FEATURE_NAMES = ["rushejiao", "sin_rushejiao", "cos_rushejiao",
                    "new_Sigma0_VH_db", "new_Sigma0_VV_db",
                    "VH+VV", "VH-VV", "VH_VV_division", "VH_VV_multiply"]
# S2 链 11 特征（S2chuli.xml BandMaths 导出，裁剪后带 _masked 后缀）
S2_FEATURE_NAMES = ["NDVI", "NDMI", "RVI", "NDII", "MSI", "WBI", "FVI",
                    "Red", "NIR", "Swir1", "Swir2"]
S1_CLIP_FILES = [n + ".tif" for n in S1_FEATURE_NAMES]
S2_CLIP_FILES = [n + "_masked.tif" for n in S2_FEATURE_NAMES]

SOIL_RANGE_HARD = (0.0, 1.0)   # 相对湿度物理硬界
SOIL_RANGE_SOFT = (0.2, 0.45)  # 训练目标 ~0.31 的合理邻域（超界仅提示）


class SoilRangeError(RuntimeError):
    """反演湿度超出物理硬界 [0,1] 时抛出（防旧品式无声失真）。"""


# ============================================================
# 辅助函数
# ============================================================
def _list_tif(folder):
    """返回文件夹内 .tif 文件名列表（Python sorted 序——训练采样用的就是它）。"""
    return sorted(f for f in os.listdir(folder) if f.endswith(".tif"))


def _count_files(folder, suffix=None):
    if not os.path.isdir(folder):
        return 0
    names = os.listdir(folder)
    if suffix:
        names = [n for n in names if n.endswith(suffix)]
    return len(names)


def _count_tif_deep(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for _, _, files in os.walk(folder) for f in files if f.endswith(".tif"))


def _run_gpt(xml_file, zip_path, output_file):
    """执行 gpt 图（旧脚本 1.py 390 行 / S2.py 520 行的 -Pinput/-Poutput 参数化方式）。"""
    cmd = [GPT_EXE, xml_file, f"-Pinput={zip_path}", f"-Poutput={output_file}"]
    t0 = time.time()
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=GPT_TIMEOUT)
    except subprocess.CalledProcessError as e:
        out = (e.output or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"gpt 失败（exit={e.returncode}）：{out[-800:]}") from e
    return round(time.time() - t0, 1)


# ============================================================
# 工具 1：阶段状态盘查
# ============================================================
def soil_inspect_data(base_dir=DEFAULT_BASE_DIR, area=DEFAULT_AREA, date=None):
    """盘查某区域土壤湿度链各阶段状态（旧 test/ 与新 work/ 均查，只读）。

    返回各日期下 S1/S2 链（data_0→yuchuli→zhuan_tif_2→clip）、feature、result 的存在性，
    以及 soim_values 样本 CSV 清单。用于 Agent 判断"该日期跑到了哪一步、缺什么"。
    """
    stages = {}
    for tree in ("test", "work"):
        area_root = os.path.join(base_dir, tree, area)
        dates = []
        if os.path.isdir(area_root):
            if date:
                dates = [date] if os.path.isdir(os.path.join(area_root, date)) else []
            else:
                dates = sorted(d for d in os.listdir(area_root)
                               if os.path.isdir(os.path.join(area_root, d)))
        info = {}
        for d in dates:
            root = os.path.join(area_root, d)
            s1 = os.path.join(root, "S1")
            s2 = os.path.join(root, "S2")
            info[d] = {
                "S1": {
                    "data_0_zip": _count_files(os.path.join(s1, "data_0"), ".zip"),
                    "yuchuli_dim": _count_files(os.path.join(s1, "S1_yuchuli_1"), ".dim"),
                    "zhuan_tif_2_tif": _count_tif_deep(os.path.join(s1, "zhuan_tif_2")),
                    "clip_tif": _count_files(os.path.join(s1, "clip"), ".tif"),
                },
                "S2": {
                    "data_0_zip": _count_files(os.path.join(s2, "data_0"), ".zip"),
                    "yuchuli_dim": _count_files(os.path.join(s2, "S2_yuchuli_1"), ".dim"),
                    "zhuan_tif_2_tif": _count_tif_deep(os.path.join(s2, "zhuan_tif_2")),
                    "mosaic_3_file": _count_tif_deep(os.path.join(s2, "mosaic_3")),
                    "clip_tif": _count_files(os.path.join(s2, "clip"), ".tif"),
                },
                "feature_tif": _count_files(os.path.join(root, "feature"), ".tif"),
                "result_files": (sorted(os.listdir(os.path.join(root, "result")))
                                 if os.path.isdir(os.path.join(root, "result")) else []),
            }
        stages[tree] = info

    csvs = []
    csv_dir = os.path.join(base_dir, "test", area, "soim_values")
    if os.path.isdir(csv_dir):
        for f in sorted(os.listdir(csv_dir)):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(csv_dir, f))
                    csvs.append({"file": f, "rows": int(len(df)), "columns": list(df.columns)})
                except Exception:
                    csvs.append({"file": f, "rows": None, "columns": None})

    hint = ("完整链要求：data_0 有 zip → gpt 出 yuchuli .dim → zhuan_tif_2 转 tif → clip 裁剪 → "
            "feature 装配 → 训练 → 反演。样本 CSV 无对应影像的日期不可用（特征无处采样）。")
    return _json_safe({"area": area, "stages": stages, "csvs": csvs, "hint": hint})


# ============================================================
# 工具 2/3：S1 链（gpt 预处理 + Python 后处理）
# ============================================================
def soil_s1_gpt(zip_path, yuchuli_dir):
    """对单个 S1 GRDH zip 跑 gpt 哨兵1预处理.xml（旧 1.py 379-394 行），幂等。

    图内容（哨兵1预处理.xml）：热噪声去除→轨道文件→Sigma0 定标→Refined Lee 滤波→地形校正
    （WGS84 DD 10m）→LinearToFromdB→BandMaths（VH/VV 及四则组合）→导出入射角与各特征。
    输出 Subset_{zip名去扩展名}_features.dim（SNAP 同写 .data 目录）。
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"S1 zip 不存在: {zip_path}")
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    os.makedirs(yuchuli_dir, exist_ok=True)
    out_dim = os.path.join(yuchuli_dir, f"Subset_{stem}_features.dim")
    if os.path.isfile(out_dim) and os.path.isdir(out_dim.replace(".dim", ".data")):
        return {"status": "exists", "output": out_dim, "note": "yuchuli 产物已存在，跳过 gpt"}
    duration = _run_gpt(GRAPH_S1, zip_path, out_dim)
    return {"status": "success", "output": out_dim, "duration_s": duration,
            "note": "gpt 完成；下一步用 soil_s1_features 转 tif/计算 sin-cos/裁剪"}


def soil_s1_features(dim_dir, shp_path, out_tif_dir, out_clip_dir):
    """S1 yuchuli .dim → 9 个特征 tif（旧 1.py img_to_tif/calculate_sin_cos/crop_rasters），幂等。

    dim_dir 可以是旧 yuchuli（只读复用，51208 验证用）或新 gpt 输出。
    三步逐字复刻旧脚本：features.data/*.img→tif（1.py 172-183 行）→ rushejiao 非零像元
    sin/cos（185-234 行，正则缺陷已修，见模块 docstring）→ shp 裁剪 0→NaN（278-321 行）。
    """
    if not os.path.isdir(dim_dir):
        raise FileNotFoundError(f"yuchuli 目录不存在: {dim_dir}")
    converted, sin_cos, cropped = [], [], []
    shapefile = gpd_read(shp_path)
    geometries = [g for g in shapefile.geometry]  # 旧 1.py 290-291 行

    for entry in sorted(os.listdir(dim_dir)):
        if not entry.endswith("features.data"):  # 旧 1.py 160-161 行
            continue
        img_folder = os.path.join(dim_dir, entry)
        tif_folder = os.path.join(out_tif_dir, entry)
        os.makedirs(tif_folder, exist_ok=True)

        # --- 段1：img_to_tif（旧 1.py 172-183 行）---
        for img_file in sorted(glob.glob(os.path.join(img_folder, "*.img"))):
            tif_file = os.path.join(tif_folder, os.path.basename(img_file).replace(".img", ".tif"))
            if os.path.isfile(tif_file):
                converted.append({"file": os.path.basename(tif_file), "status": "exists"})
                continue
            with rasterio.open(img_file) as src:
                with rasterio.open(tif_file, "w", driver="GTiff", count=src.count,
                                   dtype=src.dtypes[0], crs=src.crs, transform=src.transform,
                                   width=src.width, height=src.height) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
            converted.append({"file": os.path.basename(tif_file), "status": "success"})

        # --- 段2：calculate_sin_cos（旧 1.py 185-234 行，仅精确匹配 rushejiao.tif）---
        rushe = os.path.join(tif_folder, "rushejiao.tif")
        if os.path.isfile(rushe):
            sin_out = rushe.replace("rushejiao.tif", "sin_rushejiao.tif")  # 旧 217-218 行
            cos_out = rushe.replace("rushejiao.tif", "cos_rushejiao.tif")
            if os.path.isfile(sin_out) and os.path.isfile(cos_out):
                sin_cos.append({"base": "rushejiao", "status": "exists"})
            else:
                with rasterio.open(rushe) as src:
                    img = src.read(1)
                    non_zero = img != 0  # 旧 207 行：仅非零像元计算
                    sin_data = np.zeros_like(img, dtype=np.float32)
                    cos_data = np.zeros_like(img, dtype=np.float32)
                    sin_data[non_zero] = np.sin(np.radians(img[non_zero]))
                    cos_data[non_zero] = np.cos(np.radians(img[non_zero]))
                    for out_path, arr in ((sin_out, sin_data), (cos_out, cos_data)):
                        with rasterio.open(out_path, "w", driver="GTiff", count=1, dtype=arr.dtype,
                                           crs=src.crs, transform=src.transform,
                                           width=src.width, height=src.height) as dst:
                            dst.write(arr, 1)
                sin_cos.append({"base": "rushejiao", "status": "success"})

        # --- 段3：crop_rasters（旧 1.py 278-321 行）---
        os.makedirs(out_clip_dir, exist_ok=True)
        for file in sorted(os.listdir(tif_folder)):
            if not file.endswith(".tif"):
                continue
            input_path = os.path.join(tif_folder, file)
            output_path = os.path.join(out_clip_dir, file.replace("_merge.tif", "_cropped.tif"))
            if os.path.isfile(output_path):
                cropped.append({"file": os.path.basename(output_path), "status": "exists"})
                continue
            with rasterio.open(input_path) as src:
                out_meta = src.meta.copy()
                out_image, out_transform = mask(src, geometries, crop=True)
                img_data = np.where(out_image == 0, np.nan, out_image)  # 旧 306 行：0→NaN
                out_meta.update({"driver": "GTiff", "height": img_data.shape[1],
                                 "width": img_data.shape[2], "transform": out_transform})
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(img_data)
            cropped.append({"file": os.path.basename(output_path), "status": "success"})

    # 护栏：产物清单与物理界检查
    warns = []
    clip_files = _list_tif(out_clip_dir)
    missing = [f for f in S1_CLIP_FILES if f not in clip_files]
    if missing:
        warns.append(f"S1 clip 缺特征: {missing}")
    if os.path.isfile(os.path.join(out_clip_dir, "sin_rushejiao.tif")):
        with rasterio.open(os.path.join(out_clip_dir, "sin_rushejiao.tif")) as src:
            vals = src.read(1)
            v = vals[~np.isnan(vals)]
            if v.size and (v.min() < -1.0 or v.max() > 1.0):
                warns.append(f"sin_rushejiao 超 [-1,1]: {float(v.min()):.4f}~{float(v.max()):.4f}")
    for db_name in ("new_Sigma0_VH_db", "new_Sigma0_VV_db"):
        p = os.path.join(out_clip_dir, db_name + ".tif")
        if os.path.isfile(p):
            with rasterio.open(p) as src:
                vals = src.read(1)
                v = vals[~np.isnan(vals)]
                if v.size and (v.min() < -50 or v.max() > 30):
                    warns.append(f"{db_name} 超出常见 dB 域 [-50,30]: {float(v.min()):.2f}~{float(v.max()):.2f}")

    return _json_safe({"converted": converted, "sin_cos": sin_cos, "cropped": cropped,
                       "clip_tif_count": len(clip_files), "clip_files": clip_files,
                       "expected": len(S1_CLIP_FILES), "warns": warns,
                       "note": "S1 特征完成；下一步 soil_assemble_features 与 S2 合并"})


# ============================================================
# 工具 4/5：S2 链（gpt 预处理 + Python 后处理）
# ============================================================
def soil_s2_gpt(zip_path, yuchuli_dir):
    """对单个 S2 L2A zip 跑 gpt S2chuli.xml（旧 S2.py 514-522 行），幂等。

    图内容（S2chuli.xml）：pixelRegion(0,0,10980,10980)→20m 重采样（上采样双线性/下采样均值）
    →11 个 BandMaths（NDVI/NDMI/RVI/NDII/MSI/WBI/FVI/Red/NIR/Swir1/Swir2）→BandMerge→BEAM-DIMAP。
    输出 {zip名去扩展名}.dim。注意：S2 过境日可能≠样本日期（如 60513 的影像为 0514 过境）。
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"S2 zip 不存在: {zip_path}")
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    os.makedirs(yuchuli_dir, exist_ok=True)
    out_dim = os.path.join(yuchuli_dir, stem + ".dim")
    if os.path.isfile(out_dim) and os.path.isdir(out_dim.replace(".dim", ".data")):
        return {"status": "exists", "output": out_dim, "note": "yuchuli 产物已存在，跳过 gpt"}
    duration = _run_gpt(GRAPH_S2, zip_path, out_dim)
    return {"status": "success", "output": out_dim, "duration_s": duration,
            "note": "gpt 完成；下一步用 soil_s2_features 转 tif/镶嵌/重投影裁剪"}


def _reproject_and_crop(input_raster, geometries, target_crs, output_raster):
    """单文件重投影+裁剪（旧 S2.py 399-463 行复刻）。

    修复点：旧代码 rasterio.open("mem", "w+") 本意用 GDAL MEM 内存驱动，但 Windows 上
    GDAL 按文件名落盘成真实 GTiff（117MB 垃圾文件）；本工具改 MemoryFile 纯内存，数值语义不变。
    """
    with rasterio.open(input_raster) as src:
        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({"crs": target_crs, "transform": transform,
                           "width": width, "height": height})
            with MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                                  src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=transform, dst_crs=target_crs,
                                  resampling=Resampling.nearest)  # 旧 435 行：nearest
                    out_image, out_transform = mask(dst, geometries, crop=True, filled=False)
                    out_meta = dst.meta.copy()
        else:
            out_image, out_transform = mask(src, geometries, crop=True, filled=False)
            out_meta = src.meta.copy()

        out_image = out_image.astype(np.float32)
        out_image[out_image.mask] = np.nan
        out_image = out_image.data

        out_meta.update({"driver": "GTiff", "height": out_image.shape[1],
                         "width": out_image.shape[2], "transform": out_transform,
                         "dtype": "float32", "nodata": np.nan})
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)


def _process_tif_files(input_folder, geometries, target_crs, output_folder):
    """文件夹内 .tif 逐个 _masked.tif 裁剪（旧 S2.py 465-485 行复刻）。"""
    os.makedirs(output_folder, exist_ok=True)
    done = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".tif"):
                input_tif = os.path.join(root, file)
                output_tif = os.path.join(output_folder, file.replace(".tif", "_masked.tif"))
                if os.path.isfile(output_tif):
                    done.append({"file": os.path.basename(output_tif), "status": "exists"})
                    continue
                _reproject_and_crop(input_tif, geometries, target_crs, output_tif)
                done.append({"file": os.path.basename(output_tif), "status": "success"})
    return done


def soil_s2_features(safe_dir, shp_path, out_tif_dir, out_mosaic_dir, out_clip_dir):
    """S2 yuchuli .dim → 11 个 _masked.tif（旧 S2.py 各段复刻），幂等。

    三步：SAFE.data/*.img→tif（173-200 行）→ 按文件夹名日期分组镶嵌（214-323 行，
    单景跳过——我们的日期均为单景，mosaic_3 为空属正常）→ 重投影到 shp CRS 并裁剪
    （399-485 行，nearest 重投影 + filled=False 背景 NaN）。裁剪分两支：mosaic_3 子目录
    >1 时按日期子目录裁，否则直接从 zhuan_tif_2 子目录裁（旧 554-570 行逐字保留）。
    """
    if not os.path.isdir(safe_dir):
        raise FileNotFoundError(f"yuchuli 目录不存在: {safe_dir}")
    shapefile = gpd_read(shp_path)
    target_crs = shapefile.crs
    geometries = [f["geometry"] for f in shapefile.__geo_interface__["features"]]  # 旧 407-408 行
    converted = []

    # --- 段1：img_to_tif（旧 S2.py 173-200 行，过滤 SAFE.data）---
    for entry in sorted(os.listdir(safe_dir)):
        if not entry.endswith("SAFE.data"):
            continue
        img_folder = os.path.join(safe_dir, entry)
        tif_folder = os.path.join(out_tif_dir, entry)
        os.makedirs(tif_folder, exist_ok=True)
        for img_file in sorted(glob.glob(os.path.join(img_folder, "*.img"))):
            tif_file = os.path.join(tif_folder, os.path.basename(img_file).replace(".img", ".tif"))
            if os.path.isfile(tif_file):
                converted.append({"file": os.path.basename(tif_file), "status": "exists"})
                continue
            with rasterio.open(img_file) as src:
                with rasterio.open(tif_file, "w", driver="GTiff", count=src.count,
                                   dtype=src.dtypes[0], crs=src.crs, transform=src.transform,
                                   width=src.width, height=src.height) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
            converted.append({"file": os.path.basename(tif_file), "status": "success"})

    # --- 段2：按日期分组镶嵌（旧 S2.py 214-323 行；单文件组跳过=不产出）---
    date_groups = {}
    for root, dirs, _ in os.walk(out_tif_dir):
        for dir_name in dirs:
            full = os.path.join(root, dir_name)
            date_str = None
            if len(dir_name) >= 19:
                cand = dir_name[11:19]  # 旧 208 行：文件夹名第12-19字符
                if cand.isdigit() and len(cand) == 8:
                    date_str = cand
            if date_str:
                date_groups.setdefault(date_str, {})
                for tif_file in sorted(glob.glob(os.path.join(full, "*.tif"))):
                    date_groups[date_str].setdefault(os.path.basename(tif_file), []).append(tif_file)

    mosaicked = []
    for date_str, file_dict in date_groups.items():
        date_out = os.path.join(out_mosaic_dir, date_str)
        for file_name, file_paths in file_dict.items():
            if len(file_paths) == 1:
                continue  # 旧 307-309 行：单文件跳过镶嵌
            os.makedirs(date_out, exist_ok=True)
            output_file = os.path.join(date_out, f"mosaic_{file_name}")  # 旧 314 行
            if os.path.isfile(output_file):
                mosaicked.append({"file": f"mosaic_{file_name}", "status": "exists"})
                continue
            srcs = [rasterio.open(p) for p in file_paths]
            mosaic_arr, out_trans = merge(srcs)
            out_meta = srcs[0].meta.copy()
            out_meta.update({"height": mosaic_arr.shape[1], "width": mosaic_arr.shape[2],
                             "transform": out_trans, "compress": "lzw", "tiled": True})  # 旧 258-265 行
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic_arr)
            for s in srcs:
                s.close()
            mosaicked.append({"file": f"mosaic_{file_name}", "status": "success"})

    # --- 段3：裁剪（旧 S2.py 554-570 行两分支逐字保留）---
    cropped = []
    subdir_count = sum(1 for e in os.scandir(out_mosaic_dir) if e.is_dir()) if os.path.isdir(out_mosaic_dir) else 0
    if subdir_count > 1:
        for filename in os.listdir(out_mosaic_dir):
            input_folder = os.path.join(out_mosaic_dir, filename)
            output_folder = os.path.join(out_clip_dir, filename)
            cropped += _process_tif_files(input_folder, geometries, target_crs, output_folder)
    else:
        for filename in os.listdir(out_tif_dir):
            input_folder = os.path.join(out_tif_dir, filename)
            cropped += _process_tif_files(input_folder, geometries, target_crs, out_clip_dir)

    clip_files = _list_tif(out_clip_dir)
    missing = [f for f in S2_CLIP_FILES if f not in clip_files]
    warns = [f"S2 clip 缺特征: {missing}"] if missing else []
    if os.path.isfile(os.path.join(out_clip_dir, "NDVI_masked.tif")):
        with rasterio.open(os.path.join(out_clip_dir, "NDVI_masked.tif")) as src:
            vals = src.read(1)
            v = vals[~np.isnan(vals)]
            if v.size and (v.min() < -1.0 or v.max() > 1.0):
                warns.append(f"NDVI 超 [-1,1]: {float(v.min()):.4f}~{float(v.max()):.4f}")

    return _json_safe({"converted": converted, "mosaicked": mosaicked, "cropped": cropped,
                       "clip_tif_count": len(clip_files), "clip_files": clip_files,
                       "expected": len(S2_CLIP_FILES), "warns": warns,
                       "note": "S2 特征完成；下一步 soil_assemble_features 与 S1 合并"})


# ============================================================
# 工具 6：特征装配（复制 + 采样提取）
# ============================================================
def soil_assemble_features(s1_clip, s2_clip, csv_path, out_feature_dir, out_csv):
    """合并 S1(9)+S2(11) 裁剪特征并提取样本特征值（旧 RF_test.py 80-171 行），幂等。

    复制（18-91 行，含重名计数器）→ 若 out_csv 已存在直接加载（旧 102-116 行行为），
    否则按 sorted 文件名顺序逐像元采样（96-171 行逐字复刻：src.index 取像元、越界/NaN→NaN、
    全特征有效才保留），写 feature_target.csv（列 Feature_1..N + Target，index=False）。
    """
    # --- 复制（旧 RF_test.py 18-91 行）---
    if not os.path.isdir(s1_clip):
        raise FileNotFoundError(f"S1 clip 不存在: {s1_clip}")
    if not os.path.isdir(s2_clip):
        raise FileNotFoundError(f"S2 clip 不存在: {s2_clip}")
    os.makedirs(out_feature_dir, exist_ok=True)
    copied = 0
    for source_folder in (s1_clip, s2_clip):
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(".tif"):
                source_file = os.path.join(source_folder, filename)
                destination_file = os.path.join(out_feature_dir, filename)
                # 旧脚本（RF_test.py 44-47 行）重名时追加 _N 计数改名——但干净运行 S1/S2
                # 文件名不重叠，计数器只在工具重跑时把产物复制成 *_1.tif 造成特征翻倍污染，
                # 故改为 exists 跳过（单次运行语义与旧脚本完全一致，重跑安全，与火点链同口径）。
                if os.path.exists(destination_file):
                    continue
                shutil.copy2(source_file, destination_file)
                copied += 1

    # --- 采样（旧 RF_test.py 96-171 行逐字复刻）---
    data = pd.read_csv(csv_path)
    coordinates = data.iloc[:, :2].values
    target = data.iloc[:, -1].values

    if os.path.exists(out_csv):
        feature_data = pd.read_csv(out_csv)
        features = feature_data.iloc[:, :-1].values
        target = feature_data.iloc[:, -1].values
        valid_mask = ~np.any(np.isnan(features), axis=1)
        if not np.all(valid_mask):
            features = features[valid_mask]
            target = target[valid_mask]
        return _json_safe({"status": "exists", "copied": copied,
                           "raw_samples": int(len(data)),
                           "valid_samples": int(len(features)),
                           "feature_columns": int(features.shape[1]),
                           "csv": out_csv,
                           "note": "feature_target.csv 已存在，直接加载（旧脚本同款行为）"})

    tif_files = [os.path.join(out_feature_dir, f)
                 for f in os.listdir(out_feature_dir) if f.endswith(".tif")]
    tif_files.sort()  # 旧 121 行：确保顺序一致（训练列序 = Python sorted 序）
    feature_stack = []
    for tif in tif_files:
        with rasterio.open(tif) as src:
            grid_data = src.read(1)
            feature_values = []
            for x, y in coordinates:
                py, px = src.index(x, y)  # 旧 133 行：rasterio 返回 int 行列号
                if 0 <= px < src.width and 0 <= py < src.height:
                    value = grid_data[py, px]
                    if value != src.nodata and not np.isnan(value):
                        feature_values.append(value)
                    else:
                        feature_values.append(np.nan)
                else:
                    feature_values.append(np.nan)
            feature_stack.append(feature_values)

    features = np.array(feature_stack).T
    valid_mask = ~np.any(np.isnan(features), axis=1)
    features_valid = features[valid_mask]
    target_valid = target[valid_mask]
    if len(features_valid) == 0:
        raise ValueError("错误：没有找到任何所有特征都有有效值的样本！")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    feature_target_df = pd.DataFrame(features_valid,
                                     columns=[f"Feature_{i + 1}" for i in range(features_valid.shape[1])])
    feature_target_df["Target"] = target_valid
    feature_target_df.to_csv(out_csv, index=False)

    return _json_safe({"status": "success", "copied": copied,
                       "raw_samples": int(len(features)), "valid_samples": int(len(features_valid)),
                       "feature_columns": int(features_valid.shape[1]),
                       "feature_order": [os.path.basename(t) for t in tif_files],
                       "csv": out_csv, "note": "采样完成；下一步 soil_train_model"})


# ============================================================
# 工具 7：随机森林训练（+ joblib 持久化）
# ============================================================
def soil_train_model(feature_csv, out_dir, feature_dir=None, force=False):
    """训练随机森林模型并保存 joblib + 指标 JSON（旧 RF_test.py 174-268 行逐字复刻）。

    复刻要点（均为旧语义，不得"顺手修正"）：
    - GridSearchCV 在**未缩放**的 80% 训练切分（train_test_split rs=42）上搜索；参数网格
      3×3×4×3×3，cv=5，scoring=neg_mean_squared_error，n_jobs=-1
    - MinMaxScaler 对 X、y 均只在训练切分上 fit，再变换全量
    - 五折 CV（KFold shuffle rs=42）在**缩放后**特征上评估 RMSE/R²（反归一化后计算）
    - 最终模型 = best_params + random_state=42 在全量缩放特征上 fit
    - 特征选择 threshold=0（旧 174-185 行）→ 恒全选，故直接用全部列

    保存：{out_dir}/model.joblib（模型+缩放器+特征名+参数）与 metrics.json（CV 指标/重要性）。
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, train_test_split
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(feature_csv)
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    n_samples, n_features = features.shape

    # 特征名 = feature_dir 的 sorted 文件名（与采样列序一致；与旧 CSV 的 Feature_N 对应）
    feature_names = _list_tif(feature_dir) if (feature_dir and os.path.isdir(feature_dir)) else [
        f"Feature_{i + 1}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError(f"特征名数 {len(feature_names)} ≠ 列数 {n_features}，feature_dir 与 csv 不一致")

    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, "model.joblib")
    metrics_file = os.path.join(out_dir, "metrics.json")
    if not force and os.path.isfile(model_file) and os.path.isfile(metrics_file):
        with open(metrics_file, encoding="utf-8") as f:
            m = json.load(f)
        return _json_safe({"status": "exists", "model_file": model_file, "metrics_file": metrics_file,
                           "best_params": m.get("best_params"),
                           "cv_rmse_mean": m.get("cv_rmse_mean"), "cv_r2_mean": m.get("cv_r2_mean"),
                           "note": "模型产物已存在，跳过训练（force=True 可强制重训）"})

    X, y = features, target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 旧 198 行

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_train)  # 旧 200-202 行：只在训练集上拟合
    scaler_y.fit(y_train.reshape(-1, 1))
    features_scaled = scaler_X.transform(X)
    target_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    param_grid = {  # 旧 209-215 行
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)  # 旧 217-218 行：注意是未缩放 X_train
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []
    for train_index, test_index in kf.split(features_scaled):
        m = RandomForestRegressor(**best_params, random_state=42)
        m.fit(features_scaled[train_index], target_scaled[train_index])
        predictions_scaled = m.predict(features_scaled[test_index])
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        y_test_original = scaler_y.inverse_transform(target_scaled[test_index].reshape(-1, 1)).ravel()
        rmse_list.append(float(np.sqrt(mean_squared_error(y_test_original, predictions))))
        r2_list.append(float(r2_score(y_test_original, predictions)))

    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(features_scaled, target_scaled)

    import joblib
    joblib.dump({"model": final_model, "scaler_X": scaler_X, "scaler_y": scaler_y,
                 "feature_names": feature_names, "best_params": best_params,
                 "n_samples": int(n_samples), "n_features": int(n_features)}, model_file)
    metrics = {
        "n_samples": int(n_samples), "n_features": int(n_features),
        "feature_names": feature_names,
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_rmse_mean": float(np.mean(rmse_list)), "cv_rmse_std": float(np.std(rmse_list)),
        "cv_r2_mean": float(np.mean(r2_list)), "cv_r2_std": float(np.std(r2_list)),
        "cv_rmse_folds": rmse_list, "cv_r2_folds": r2_list,
        "feature_importances": [float(v) for v in best_model.feature_importances_],
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return _json_safe({"status": "success", "model_file": model_file, "metrics_file": metrics_file,
                       "n_samples": metrics["n_samples"], "n_features": metrics["n_features"],
                       "best_params": best_params, "cv_rmse_mean": metrics["cv_rmse_mean"],
                       "cv_r2_mean": metrics["cv_r2_mean"],
                       "note": "训练完成；下一步 soil_predict_map（默认 correct 模式）"})


# ============================================================
# 工具 8：反演预测（correct / legacy 双模式）
# ============================================================
def _predict_valid(flat_features, model, scaler_X, scaler_y):
    """共享预测内核（旧 RF_test.py 302-317 行语义）：NaN 掩膜→缩放→预测→反归一化。"""
    nan_mask = np.any(np.isnan(flat_features), axis=1)
    valid_features = flat_features[~nan_mask]
    if valid_features.size == 0:
        raise ValueError("所有输入特征包含NaN值，无法进行预测。")
    valid_scaled = scaler_X.transform(valid_features)
    valid_pred = scaler_y.inverse_transform(model.predict(valid_scaled).reshape(-1, 1)).ravel()
    predictions = np.full(flat_features.shape[0], np.nan, dtype=np.float32)
    predictions[~nan_mask] = valid_pred
    return predictions


def soil_predict_map(feature_dir, model_file, out_tif, mode="correct", force=False):
    """用已训练模型反演土壤湿度分布图（旧 RF_test.py 272-341 行），双模式。

    mode="correct"（默认，修复两处旧品 bug）：
        - 特征列按模型训练时保存的 feature_names（= sorted 序）对齐，不再用 listdir 序
        - 网格统一到 S1 10m 参考格（rushejiao.tif），S2 20m 层用 rasterio.warp.reproject
          双线性重投影（带地理对齐），不再 out_shape 纯数组拉伸
        - 分块预测（256 行/块）控制内存；输出 float32、无 nodata 元数据（同旧口径，NaN 即无效）
    mode="legacy"（验证用，逐字复现旧品）：
        - listdir 序堆叠 + 形状不符 out_shape 双线性拉伸 + np.resize 兜底
        - 与旧 soil_moisture_map.tif 在相同模型下应 bit-exact（同机同版 sklearn）

    护栏：反演值超出 [0,1] 物理硬界抛 SoilRangeError；超出 [0.2,0.45] 软界仅提示。
    """
    import joblib
    if mode not in ("correct", "legacy"):
        raise ValueError(f"mode 必须为 correct 或 legacy，收到: {mode}")
    os.makedirs(os.path.dirname(os.path.abspath(out_tif)), exist_ok=True)
    if not force and os.path.isfile(out_tif):
        return _json_safe({"status": "exists", "mode": mode, "out_tif": out_tif,
                           "note": "预测产物已存在，跳过（force=True 可强制重预测）"})
    bundle = joblib.load(model_file)
    model = bundle["model"]
    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    feature_names = list(bundle.get("feature_names", []))

    if mode == "legacy":
        # —— 旧 predict_soil_moisture（RF_test.py 273-341 行）逐字复刻 ——
        tif_files = [os.path.join(feature_dir, f)
                     for f in os.listdir(feature_dir) if f.endswith(".tif")]  # 273 行：listdir 序
        feature_stack = []
        target_shape = None
        crs = None
        transform = None
        for tif in tif_files:
            with rasterio.open(tif) as src:
                data = src.read(1)
                if target_shape is None:
                    target_shape = data.shape
                    crs = src.crs
                    transform = src.transform
                elif data.shape != target_shape:
                    data = np.array(src.read(1, out_shape=(target_shape[0], target_shape[1]),
                                             resampling=rasterio.enums.Resampling.bilinear))  # 289-292 行
                feature_stack.append(data)
        feature_stack = [np.resize(layer, target_shape) for layer in feature_stack]  # 296 行
        features = np.stack(feature_stack, axis=-1)
        rows, cols, _ = features.shape
        flat = features.reshape(-1, features.shape[2])
        predictions = _predict_valid(flat, model, scaler_X, scaler_y)
        predictions_reshaped = predictions.reshape(rows, cols)
        with rasterio.open(out_tif, "w", driver="GTiff", height=rows, width=cols,
                           count=1, dtype=np.float32, crs=crs, transform=transform) as dst:
            dst.write(predictions_reshaped, 1)
        ref_shape, ref_crs, ref_transform = (rows, cols), crs, transform
    else:
        # —— correct 模式：列序对齐 + 正确重投影 + 分块预测 ——
        if len(feature_names) == 0:
            feature_names = [os.path.basename(t) for t in
                             sorted(os.path.join(feature_dir, f)
                                    for f in os.listdir(feature_dir) if f.endswith(".tif"))]
        missing = [n for n in feature_names if not os.path.isfile(os.path.join(feature_dir, n))]
        if missing:
            raise FileNotFoundError(f"feature 目录缺训练特征: {missing}")

        ref_name = "rushejiao.tif" if "rushejiao.tif" in feature_names else feature_names[0]
        with rasterio.open(os.path.join(feature_dir, ref_name)) as ref_src:
            ref_shape = ref_src.shape
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform

        # S2 层一次性重投影到参考格（先全图重投影，再分块预测）
        layers = []
        for name in feature_names:
            with rasterio.open(os.path.join(feature_dir, name)) as src:
                data = src.read(1)
                if data.shape == ref_shape and src.transform == ref_transform:
                    layers.append(data)
                else:
                    warped = np.empty(ref_shape, dtype=np.float32)
                    reproject(source=data, destination=warped,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=ref_transform, dst_crs=ref_crs,
                              resampling=Resampling.bilinear,
                              src_nodata=np.nan, dst_nodata=np.nan)
                    layers.append(warped)

        rows, cols = ref_shape
        predictions = np.empty((rows, cols), dtype=np.float32)
        block = 256
        for r0 in range(0, rows, block):
            r1 = min(r0 + block, rows)
            flat = np.stack([layer[r0:r1, :] for layer in layers], axis=-1).reshape(-1, len(layers))
            predictions[r0:r1, :] = _predict_valid(flat, model, scaler_X, scaler_y).reshape(r1 - r0, cols)
        with rasterio.open(out_tif, "w", driver="GTiff", height=rows, width=cols,
                           count=1, dtype=np.float32, crs=ref_crs, transform=ref_transform) as dst:
            dst.write(predictions, 1)

    vals = predictions[~np.isnan(predictions)]
    v_min, v_max = (float(vals.min()), float(vals.max())) if vals.size else (None, None)
    nan_ratio = float(np.isnan(predictions).mean())
    warns = []
    if vals.size:
        if v_min < SOIL_RANGE_HARD[0] or v_max > SOIL_RANGE_HARD[1]:
            raise SoilRangeError(f"反演湿度超出物理硬界 [0,1]: {v_min:.4f}~{v_max:.4f}")
        if v_min < SOIL_RANGE_SOFT[0] or v_max > SOIL_RANGE_SOFT[1]:
            warns.append(f"超出训练目标邻域 [0.2,0.45]: {v_min:.4f}~{v_max:.4f}（提示，不阻断）")

    return _json_safe({"status": "success", "mode": mode, "out_tif": out_tif,
                       "shape": list(ref_shape), "crs": str(ref_crs),
                       "transform": list(ref_transform)[:6],
                       "value_min": v_min, "value_max": v_max, "nan_ratio": nan_ratio,
                       "warns": warns,
                       "note": ("correct=修复列序与重投影后的科学版；legacy=逐字复现旧品（含双 bug），"
                                "仅用于验证对照")})


# ============================================================
# 工具 9：模型质量自动评估
# ============================================================
def soil_assess_model(model_file, feature_csv, product_tif=None, sample_csv=None):
    """模型质量自动评估：CV 指标复算 + 特征重要性 + 样本点散点指标。

    - CV：用保存的缩放器与 best_params 复算五折 RMSE/R²（与训练时同 seed 同过程，结果应一致）
    - 特征重要性：从保存模型读取，按降序排列
    - 样本点检验（可选）：product_tif 在 sample_csv 坐标处取值，与样本值算 R²/RMSE/bias。
      口径须知：本项目样本为随机模拟数据（无真实实测值），且模型在全部样本上训练，
      此检验属样本内回代——R² 高值多为对噪声的过拟合，只能作 bug 修复方向的机械对照，
      不作为精度证据；模型能力的诚实估计只看 CV（随机目标下 CV R²≈0 是预期而非缺陷）。
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold

    bundle = joblib.load(model_file)
    model, scaler_X, scaler_y = bundle["model"], bundle["scaler_X"], bundle["scaler_y"]
    best_params = bundle.get("best_params", {})
    feature_names = list(bundle.get("feature_names", []))

    df = pd.read_csv(feature_csv)
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    features_scaled = scaler_X.transform(features)
    target_scaled = scaler_y.transform(target.reshape(-1, 1)).ravel()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []
    for train_index, test_index in kf.split(features_scaled):
        m = RandomForestRegressor(**best_params, random_state=42)
        m.fit(features_scaled[train_index], target_scaled[train_index])
        pred = scaler_y.inverse_transform(m.predict(features_scaled[test_index]).reshape(-1, 1)).ravel()
        y_true = scaler_y.inverse_transform(target_scaled[test_index].reshape(-1, 1)).ravel()
        rmse_list.append(float(np.sqrt(mean_squared_error(y_true, pred))))
        r2_list.append(float(r2_score(y_true, pred)))

    importances = list(bundle.get("feature_importances", []))
    if not importances and hasattr(model, "feature_importances_"):
        importances = [float(v) for v in model.feature_importances_]
    ranking = sorted(zip(feature_names, importances), key=lambda kv: -kv[1]) if feature_names else []

    sample_check = None
    if product_tif and sample_csv:
        sdf = pd.read_csv(sample_csv)
        coords = sdf.iloc[:, :2].values
        truth = sdf.iloc[:, -1].values
        preds, trues = [], []
        with rasterio.open(product_tif) as src:
            for (x, y), t in zip(coords, truth):
                py, px = src.index(x, y)
                if 0 <= px < src.width and 0 <= py < src.height:
                    v = src.read(1)[py, px]
                    if not np.isnan(v):
                        preds.append(float(v))
                        trues.append(float(t))
        if len(preds) >= 10:
            sample_check = {
                "n": len(preds),
                "rmse": float(np.sqrt(mean_squared_error(trues, preds))),
                "r2": float(r2_score(trues, preds)),
                "bias": float(np.mean(np.array(preds) - np.array(trues))),
                "pred_range": [float(np.min(preds)), float(np.max(preds))],
                "truth_range": [float(np.min(trues)), float(np.max(trues))],
            }
        else:
            sample_check = {"n": len(preds), "note": "有效样本点 <10，不计算散点指标"}

    return _json_safe({
        "cv": {"rmse_mean": float(np.mean(rmse_list)), "rmse_std": float(np.std(rmse_list)),
               "r2_mean": float(np.mean(r2_list)), "r2_std": float(np.std(r2_list)),
               "rmse_folds": rmse_list, "r2_folds": r2_list},
        "feature_importance_top": ranking[:5],
        "sample_check": sample_check,
        "note": ("样本为随机模拟数据（无实测值）：CV R²≈0 是随机目标的预期天花板而非缺陷；"
                 "sample_check 为样本内回代，高 R²=过拟合噪声，仅作机械对照；"
                 "本项目评估口径=流程跑通证明反演能力，精度数字不做产品化声明")})


# ============================================================
# 兼容旧脚本的 shp 读取（geopandas 导入放顶层外的本地化以加速工具载入）
# ============================================================
def gpd_read(shp_path):
    import geopandas as gpd
    return gpd.read_file(shp_path)
