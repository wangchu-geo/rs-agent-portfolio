"""
NDVI 统计与覆盖分级工具
目标：把 NDVI 分析逻辑改造成可以被其他代码调用的函数
      （原脚本 NDVI_CLOUD_MASK.py 是一次性运行的管线，
       这里把"结果解读"部分抽出来，做成 Agent 可以调用的工具）

设计原则：
    旧状态：自然语言需求 → AI生成脚本 → 手动执行 → 结果
    新状态：自然语言需求 → Agent理解拆解 → 自动调用本模块 → 自动解读
    每一个旧脚本都是一个等待被激活的工具，本模块是第一个。
"""
import os

import numpy as np
import rasterio

# ============================================================
# 解读标准：NDVI 值的植被覆盖等级划分（遥感专业知识的落点）
# ============================================================
NDVI_LEVELS = [
    (0.6, "茂密植被", "森林、健康作物"),
    (0.3, "中等密度植被", "农田、灌木"),
    (0.1, "稀疏植被", "干旱草地、早期作物"),
    (-1.0, "裸地或建筑用地", "无植被覆盖"),
]


def classify_ndvi(value: float) -> str:
    """单个 NDVI 值 → 植被覆盖等级文字描述"""
    for threshold, name, _ in NDVI_LEVELS:
        if value >= threshold:
            return name
    return "无有效数据"


# ============================================================
# 核心工具函数：NDVI 栅格统计
# ============================================================
def calculate_ndvi_stats(ndvi_path: str, region_name: str = "") -> dict:
    """
    读取本地 NDVI 栅格（.tif），计算统计指标，返回结构化结果

    参数：
        ndvi_path   -- NDVI 栅格文件完整路径（已完成计算/去云/裁剪的产品）
        region_name -- 研究区名称（可选，用于报告生成）

    返回：
        dict：{'region_name', 'mean', 'max', 'min', 'std',
               'valid_pixel_ratio', 'dominant_level', 'coverage', 'path'}
    """
    if not os.path.exists(ndvi_path):
        raise FileNotFoundError(f"找不到NDVI文件：{ndvi_path}")

    with rasterio.open(ndvi_path) as src:
        data = src.read(1).astype(np.float32)

        # ---------- 第一步：有效像元筛选 ----------
        # 兼容两种 nodata 形式：
        #   1) 元数据里写死的 nodata 值（如裁剪后常用的 -9999）
        #   2) 直接写入的 NaN（如去云处理后的结果）
        nodata = src.nodata
        if nodata is not None:
            valid = data[data != nodata]
        else:
            valid = data[~np.isnan(data)]

        # 过滤掉超出 NDVI 理论范围 [-1, 1] 的异常值
        valid = valid[(valid >= -1.0) & (valid <= 1.0)]

        total_pixels = data.size
        valid_pixels = valid.size
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

        # ---------- 第二步：统计指标 ----------
        stats = {
            'region_name': region_name,
            'mean': float(valid.mean()),
            'max': float(valid.max()),
            'min': float(valid.min()),
            'std': float(valid.std()),
            'valid_pixel_ratio': round(valid_ratio, 4),
            'valid_pixels': int(valid_pixels),
            'total_pixels': int(total_pixels),
            'dominant_level': classify_ndvi(float(valid.mean())),
            'path': ndvi_path,
        }

        # ---------- 第三步：各覆盖等级面积占比 ----------
        # 注意区间划分：[0.6, 1] [0.3, 0.6) [0.1, 0.3) [-1, 0.1)，互不重叠
        coverage = {}
        prev_threshold = None
        for threshold, name, description in NDVI_LEVELS:
            if prev_threshold is None:
                mask = valid >= threshold          # 最高档：>= 0.6
            elif threshold == -1.0:
                mask = valid < prev_threshold      # 最低档：< 0.1
            else:
                mask = (valid >= threshold) & (valid < prev_threshold)
            ratio = float(mask.sum()) / valid_pixels if valid_pixels > 0 else 0.0
            coverage[name] = {'ratio': round(ratio, 4), 'description': description}
            prev_threshold = threshold
        stats['coverage'] = coverage

        # ---------- 第四步：数据质量提示（帮模型正确解读统计结果） ----------
        # 裁剪后的栅格是矩形，研究区边界外的背景像元会标记为 nodata，
        # 所以 valid_pixel_ratio 偏低不一定代表云污染，模型容易误判，
        # 这里把专业判断直接写进返回结果
        if valid_ratio < 0.6:
            stats['note'] = (
                "有效像元占比偏低。注意：若该栅格经过矢量裁剪，"
                "矩形范围中研究区边界外的背景像元会被标记为nodata，"
                "这是正常现象，不代表数据质量问题；统计结果仅代表研究区内部。"
            )
        else:
            stats['note'] = "有效像元占比正常，统计结果可靠。"

    return stats
