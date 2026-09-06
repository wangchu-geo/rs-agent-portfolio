"""
冒烟脚本：单景（20251228 T50SPC）走 gpt → 转tif → 去云 三段
目的：在 Agent 全流程之前，先用直接调用验证三个工具的正确性，
     特别是 SNAP 13 直接读取 .SAFE.zip 的能力（无需解压）

运行方式：
    PYTHONUTF8=1 python smoke_one_tile.py
    单景 gpt 处理约 2-6 分钟（首景含 auxdata 预热）
"""
import json

from pipeline_tools import (
    apply_cloud_mask_date,
    convert_date_to_tif,
    snap_preprocess_date,
)

XML = r"E:/YYR/NDVI/NDVI_COLUD_MASK.xml"
WORK = r"E:/YYR/NDVI/multi_date/work/20251228"
SCENE = (r"E:/YYR/NDVI/multi_date/"
         r"S2A_MSIL2A_20251228T025201_N0511_R132_T50SPC_20251228T070611.SAFE.zip")

print("== 1/3 snap_preprocess_date（单景，约 2-6 分钟，请耐心等待）==")
r1 = snap_preprocess_date([SCENE], XML, WORK + "/yuchuli")
print(json.dumps(r1, ensure_ascii=False, indent=2))

print("\n== 2/3 convert_date_to_tif ==")
r2 = convert_date_to_tif(WORK + "/yuchuli", WORK + "/tif")
print(json.dumps(r2, ensure_ascii=False, indent=2))

print("\n== 3/3 apply_cloud_mask_date ==")
r3 = apply_cloud_mask_date(WORK + "/tif", WORK + "/quyun")
print(json.dumps(r3, ensure_ascii=False, indent=2))
