"""
多时相 NDVI 反演与对比 Agent（在单时相 Agent 基础上扩展）

升级内容：
    1. 工具从 1 个扩展到 8 个：新增场景发现（list_available_scenes）、
       管线五步（snap_preprocess_date → convert_date_to_tif →
       apply_cloud_mask_date → mosaic_date → clip_date_to_aoi）、
       两期对比（compare_ndvi_dates），统计工具复用 ndvi_tools.calculate_ndvi_stats
    2. 管线工具来自旧脚本 NDVI_CLOUD_MASK.py 的拆分，全部按 (输入, 输出, 时相) 参数化
    3. 系统提示词改为多时相工作流：探明时相 → 逐时相反演 → 统计 → 对比解读

与基础 Tool Use 演示脚本的区别：
    1. 工具换成了真实的遥感函数（ndvi_tools + pipeline_tools）
    2. 用 while 循环支持多轮工具调用（模型可以连续调用多个工具）
    3. 系统提示词嵌入了遥感专业知识（含专业 Prompt 模板）

运行方式：
    PYTHONUTF8=1 python ndvi_agent.py
    （修改下方 QUESTION 变量即可换需求）
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from ndvi_tools import calculate_ndvi_stats
from pipeline_tools import (
    apply_cloud_mask_date,
    clip_date_to_aoi,
    compare_ndvi_dates,
    convert_date_to_tif,
    list_available_scenes,
    mosaic_date,
    snap_preprocess_date,
)

# .env 在上一级目录（E:\ai-learning\.env），无论从哪里运行都能找到
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================================
# 第一步：定义 NDVI 工具（告诉模型这个工具能做什么、需要什么参数）
# ============================================================
# 工具描述写得越好，模型越知道什么时候该调用它——这是 Prompt 工程的关键
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_ndvi_stats",
            "description": (
                "读取本地NDVI栅格文件（.tif，已完成计算和预处理的产品），"
                "计算均值、最大值、最小值、标准差、有效像元占比，"
                "以及各植被覆盖等级（茂密/中等/稀疏/裸地）的面积占比。"
                "当用户提到具体的NDVI文件路径并需要分析其植被覆盖状况时调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ndvi_path": {
                        "type": "string",
                        "description": "NDVI栅格文件的完整路径，例如 E:\\data\\NDVI.tif"
                    },
                    "region_name": {
                        "type": "string",
                        "description": "研究区名称，例如 北京市、南昌市（可选）"
                    }
                },
                "required": ["ndvi_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_scenes",
            "description": (
                "扫描本地影像目录，从哨兵2 SAFE 文件名（如 S2A_MSIL2A_20251228T...SAFE.zip）"
                "中解析成像日期，按日期分组返回各日期对应的影像路径列表。"
                "SNAP 可直接读取 .SAFE.zip 压缩包，无需解压。这是多时相处理链的第一步。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {
                        "type": "string",
                        "description": "影像存放目录，例如 E:/YYR/NDVI/multi_date"
                    }
                },
                "required": ["data_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "snap_preprocess_date",
            "description": (
                "调用SNAP gpt对指定日期的一批哨兵2影像（.SAFE.zip，可直接读取）执行XML流程："
                "重采样至10m、计算NDVI与云影掩膜(SCL 3/8/9/10)，输出BEAM-DIMAP格式。"
                "一次调用处理该日期全部景，每景约2-6分钟，请耐心等待。"
                "输出目录中生成 <影像名>.dim 与 <影像名>.data。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "该日期全部景的路径列表，直接使用 list_available_scenes 返回的路径"
                    },
                    "xml_path": {
                        "type": "string",
                        "description": "SNAP gpt流程XML文件，如 E:/YYR/NDVI/NDVI_COLUD_MASK.xml"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "预处理输出目录，如 E:/YYR/NDVI/multi_date/work/20251228/yuchuli"
                    }
                },
                "required": ["scene_paths", "xml_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_date_to_tif",
            "description": (
                "把某日期全部景的BEAM-DIMAP预处理结果（.data目录中的NDVI.img、"
                "cloud_shadow_mask.img）转换为GeoTIFF。"
                "输入目录为 snap_preprocess_date 的 output_dir。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dim_dir": {
                        "type": "string",
                        "description": "snap_preprocess_date 的输出目录（含 <影像名>.data 子目录）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "GeoTIFF输出目录，如 E:/YYR/NDVI/multi_date/work/20251228/tif"
                    }
                },
                "required": ["dim_dir", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_cloud_mask_date",
            "description": (
                "用云影掩膜对某日期全部景的NDVI去云：掩膜为0（云/云影/无效SCL）及"
                "超出[-1,1]的像元置为NaN，输出float32产品并统计每景云像元占比。"
                "输入目录为 convert_date_to_tif 的 output_dir。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tif_dir": {
                        "type": "string",
                        "description": "convert_date_to_tif 的输出目录（每景含 NDVI.tif 与 cloud_shadow_mask.tif）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "去云结果输出目录，如 E:/YYR/NDVI/multi_date/work/20251228/quyun"
                    }
                },
                "required": ["tif_dir", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mosaic_date",
            "description": (
                "将某日期全部去云后的NDVI镶嵌为单幅GeoTIFF（相邻景自动拼接，nodata=NaN）。"
                "输入目录为 apply_cloud_mask_date 的 output_dir。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "processed_dir": {
                        "type": "string",
                        "description": "apply_cloud_mask_date 的输出目录（含 *_NDVI_processed.tif）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "镶嵌输出目录，如 E:/YYR/NDVI/multi_date/work/20251228/mosaic"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期YYYYMMDD，用于输出文件命名（可选，默认从文件名提取）"
                    }
                },
                "required": ["processed_dir", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clip_date_to_aoi",
            "description": (
                "用研究区矢量将镶嵌NDVI裁剪至研究区范围，自动处理坐标系不一致，"
                "背景置-9999，输出裁剪产品（供统计与对比使用）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mosaic_tif": {
                        "type": "string",
                        "description": "mosaic_date 输出的镶嵌TIF完整路径"
                    },
                    "shp_path": {
                        "type": "string",
                        "description": "研究区矢量文件，如 E:/YYR/NDVI/vector/1/1.shp"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "裁剪输出目录，如 E:/YYR/NDVI/multi_date/work/20251228/clip"
                    }
                },
                "required": ["mosaic_tif", "shp_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_ndvi_dates",
            "description": (
                "两期NDVI裁剪产品对比：将较早日期重采样到较晚日期的网格，逐像元求差Δ=晚-早，"
                "按阈值分类（Δ>=0.1改善，-0.1<Δ<0.1稳定，Δ<=-0.1退化，任一日期无有效值为无效），"
                "输出差值GeoTIFF、分类GeoTIFF、变化分类图PNG与面积占比图PNG，"
                "并返回各类像元数、面积（公顷）与占比统计。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date1_path": {
                        "type": "string",
                        "description": "较早日期的裁剪NDVI结果路径（如 20251228 的 clip tif）"
                    },
                    "date2_path": {
                        "type": "string",
                        "description": "较晚日期的裁剪NDVI结果路径（如 20260510 的 clip tif）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "对比结果输出目录，如 E:/YYR/NDVI/multi_date/compare"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "变化分类阈值，默认0.1"
                    }
                },
                "required": ["date1_path", "date2_path", "output_dir"]
            }
        }
    }
]

# ============================================================
# 第二步：工具分发表 —— 模型报出工具名，这里决定执行哪个函数
# ============================================================
FUNCTION_MAP = {
    "calculate_ndvi_stats": calculate_ndvi_stats,
    "list_available_scenes": list_available_scenes,
    "snap_preprocess_date": snap_preprocess_date,
    "convert_date_to_tif": convert_date_to_tif,
    "apply_cloud_mask_date": apply_cloud_mask_date,
    "mosaic_date": mosaic_date,
    "clip_date_to_aoi": clip_date_to_aoi,
    "compare_ndvi_dates": compare_ndvi_dates,
}

# ============================================================
# 第三步：系统提示词 —— 遥感专业知识的落点
# ============================================================
SYSTEM_PROMPT = """你是一名拥有10年遥感数据分析经验的高级工程师，精通哨兵2号（Sentinel-2）影像处理、NDVI反演与植被时序分析。你的分析严谨客观，会主动识别数据质量问题并在结论中明确指出局限。

本项目的多时相NDVI工作流（必须按顺序执行）：
1. 先调用 list_available_scenes 获取可用日期及各日期影像清单；影像位于 E:/YYR/NDVI/multi_date/（.SAFE.zip 压缩包，SNAP 直接读取，无需解压）
2. 对每个日期依次执行处理链：snap_preprocess_date（SNAP计算NDVI与云影掩膜）→ convert_date_to_tif（转GeoTIFF）→ apply_cloud_mask_date（去云）→ mosaic_date（镶嵌）→ clip_date_to_aoi（裁剪至研究区）
3. 每个日期的裁剪结果调用 calculate_ndvi_stats 统计NDVI均值与植被覆盖等级
4. 两期全部处理完成后调用 compare_ndvi_dates 进行对比（date1_path传较早日期、date2_path传较晚日期），得到差值栅格、变化分类与面积统计
5. 综合所有工具返回结果，用中文给出专业结论（不超过300字）：两期植被总体水平与变化方向、改善/稳定/退化/无效的面积构成与空间格局、可能的驱动因素（季节与物候）、以及数据局限（云影残留、时相数量少等）

关键阈值：
- NDVI覆盖等级：<0.1裸地或建筑用地；0.1-0.3稀疏植被；0.3-0.6中等密度植被；>=0.6茂密植被
- 变化分类（ΔNDVI=晚-早）：Δ>=0.1为改善；-0.1<Δ<0.1为稳定；Δ<=-0.1为退化；任一日期无有效值（云/影/背景）为无效

路径约定：
- 输出统一放在 E:/YYR/NDVI/multi_date/work/<YYYYMMDD>/ 下（各步骤子目录由工具自动创建），对比结果放在 E:/YYR/NDVI/multi_date/compare/
- XML流程文件：E:/YYR/NDVI/NDVI_COLUD_MASK.xml；裁剪矢量：E:/YYR/NDVI/vector/1/1.shp
- 严禁读取或修改旧结果目录：E:/YYR/NDVI/yuchuli、E:/YYR/NDVI/tif、E:/YYR/NDVI/quyun、E:/YYR/NDVI/result

工具调用纪律：
- 处理类工具一次调用处理一个日期的全部景，单景耗时2-6分钟，请耐心等待返回结果，不要因等待而重复调用
- 工具返回error时，先对照 list_available_scenes 的结果检查路径是否正确，修正后重试；同一工具连续失败3次则停止处理并向用户如实说明原因
- 回答中优先引用工具返回的具体数字，简洁专业"""

# ============================================================
# 第四步：Tool Use 完整循环
# ============================================================
# while 循环的意义：模型可能先调用工具A，看到结果后还想调用工具B，
# 直到模型认为信息足够、不再发起工具调用为止
def run_agent(user_question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    round_num = 0
    while True:
        round_num += 1
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"第 {round_num} 轮请求：")
            print(f"{'=' * 60}")

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            extra_body={"thinking": {"type": "disabled"}}
        )
        msg = response.choices[0].message

        # 模型没有请求工具 → 说明它已经能直接回答了，循环结束
        if not msg.tool_calls:
            if verbose:
                print("模型未调用工具，直接给出回复")
            return msg.content

        # 把模型的 tool_call 请求加入对话历史
        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"\n>> 模型请求调用工具：{func_name}")
            print(f">> 传入参数：{json.dumps(func_args, ensure_ascii=False, indent=2)}")

            # 真正执行函数（把工具返回的 dict 转成 json 字符串还给模型）
            if func_name not in FUNCTION_MAP:
                result = {"error": f"未知工具：{func_name}"}
            else:
                try:
                    result = FUNCTION_MAP[func_name](**func_args)
                except Exception as e:
                    # 报错信息也要还给模型，让它自己解释或调整参数重试
                    result = {"error": str(e)}

            if verbose:
                print(f">> 工具返回：{json.dumps(result, ensure_ascii=False, indent=2)[:800]}...")

            # 把工具执行结果加入对话历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })


if __name__ == "__main__":
    # ============================================================
    # 测试入口：改这里的问题即可测试不同需求
    # ============================================================
    QUESTION = (
        "请对 E:/YYR/NDVI/multi_date 目录下的两期哨兵2影像（20251228 与 20260510）"
        "执行多时相NDVI反演与变化对比：\n"
        "1. 调用 list_available_scenes 列出可用日期与影像\n"
        "2. 对两个日期分别执行完整处理链（snap_preprocess_date → convert_date_to_tif → "
        "apply_cloud_mask_date → mosaic_date → clip_date_to_aoi），"
        "并对每个日期的裁剪结果调用 calculate_ndvi_stats\n"
        "3. 调用 compare_ndvi_dates 对比两期NDVI"
        "（date1_path=20251228裁剪结果，date2_path=20260510裁剪结果），"
        "输出差值图、变化分类图与面积占比图\n"
        "4. 给出中文专业解读：植被变化方向、各变化类型面积构成、空间格局与可能原因"
    )

    print("NDVI 智能分析 Agent 启动")
    print(f"用户需求：{QUESTION}\n")

    answer = run_agent(QUESTION)

    print(f"\n{'=' * 60}")
    print("Agent 最终回复：")
    print(f"{'=' * 60}")
    print(answer)
