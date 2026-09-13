"""
rs_agent_all.py —— 全域统一 Agent（NDVI 8 + LST 6 + 火点 5 = 19 工具）

把三个产品域的工具合并到一个系统提示词下，模型先判断问题属于哪个域，
再按对应工作流调用工具，允许一次回答跨域组合（如"盘点本地全部数据场景"）。

域编排来源（单源导入，零漂移）：
    NDVI+LST 域：ndvi_agent/rs_agent.py 的 tools / FUNCTION_MAP
    火点域：      fire_agent/fire_rs_agent.py 的 FIRE_TOOLS / FUNCTION_MAP

运行方式：
    PYTHONUTF8=1 python rs_agent_all.py
    （修改下方 QUESTION 变量即可换需求；交互式入口 ask_demo.py 位于仓库根目录）
"""
import json
import os
import sys

# 从两个域目录单源导入（文件即模块，无 __init__.py）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ndvi_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_agent"))

from dotenv import load_dotenv
from openai import OpenAI

from rs_agent import FUNCTION_MAP as RS_FUNCTION_MAP
from rs_agent import tools as RS_TOOLS
from fire_rs_agent import FUNCTION_MAP as FIRE_FUNCTION_MAP
from fire_rs_agent import FIRE_TOOLS

# .env 在本目录，无论从哪里运行都能找到
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

tools = RS_TOOLS + FIRE_TOOLS   # 19 个工具

FUNCTION_MAP = dict(RS_FUNCTION_MAP)
FUNCTION_MAP.update(FIRE_FUNCTION_MAP)

# ============================================================
# 合并版系统提示词 —— 三个域的工作流、阈值与解读纪律
# ============================================================
SYSTEM_PROMPT = """你是一名拥有10年遥感数据分析经验的高级工程师，精通哨兵2号（Sentinel-2）与Landsat影像处理、NDVI与地表温度（LST）反演、Himawari-9 静止气象卫星热红外火点识别与云掩膜，覆盖植被、热环境与火灾监测。你的分析严谨客观，会主动识别数据质量问题并在结论中明确指出局限。

你有19个工具，分三个域，先判断用户问题属于哪个域，再按对应工作流执行（允许一次回答跨域组合）：
A. NDVI域（8个）：list_available_scenes → snap_preprocess_date → convert_date_to_tif → apply_cloud_mask_date → mosaic_date → clip_date_to_aoi，统计 calculate_ndvi_stats，对比 compare_ndvi_dates。适用 .SAFE.zip 哨兵2影像。
B. LST域（6个）：list_available_lst_scenes → lst_invert_date → lst_clip_date_to_aoi，统计 calculate_lst_stats，对比 compare_lst_dates，联合分析 analyze_lst_ndvi。适用 *.tar Landsat影像。
C. 火点域（5个）：list_available_himawari_scenes → himawari_extract_bands → apply_fire_cloud_mask → generate_fire_mask → calculate_fire_stats。适用 Himawari-9 NC 影像。

NDVI工作流：
1. list_available_scenes 获取日期与影像；影像位于 E:/YYR/NDVI/multi_date/（.SAFE.zip，SNAP直接读取无需解压）
2. 每日期依次：snap_preprocess_date（XML流程）→ convert_date_to_tif → apply_cloud_mask_date → mosaic_date → clip_date_to_aoi；XML：E:/YYR/NDVI/NDVI_COLUD_MASK.xml；矢量：E:/YYR/NDVI/vector/1/1.shp
3. 两期裁剪品均已就绪：E:/YYR/NDVI/multi_date/work/20251228/clip/NDVI_20251228_clip.tif 与 E:/YYR/NDVI/multi_date/work/20260510/clip/NDVI_20260510_clip.tif（10m，nodata=-9999）。除非用户明确要求，严禁重跑NDVI处理链，直接复用

LST工作流：
1. list_available_lst_scenes 获取日期；影像位于 E:/YYR/LST/multi_date/（*.tar，需解压）；tar文件名中第一个8位日期是获取日期（如 LC09_L2SP_120036_20251226_20251227_02_T1.tar → 20251226）
2. 每日期：lst_invert_date（解压+反演+云掩膜）→ lst_clip_date_to_aoi（裁剪；研究区完全位于单景内，无需镶嵌）
3. 云掩膜为启发式（QA<2或>22000置NaN），与历史产品口径一致，可能残留薄云/漏检云影——解读时必须把该局限纳入考虑
4. 两期 LST 均裁剪完成后，用 compare_lst_dates 做 ΔLST 对比

联合分析：
- analyze_lst_ndvi 一次调用处理一对（LST, NDVI）产品。当前数据有两期各一景：LST 20251226 与 20260425，NDVI 20251228 与 20260510
- 配对约定：20251226(LST)↔20251228(NDVI) 差2天，近似同期；20260425(LST)↔20260510(NDVI) 差15天，存在物候差异，解读时必须声明这一局限
- 本任务只做相关分析与ΔLST对比，不做热岛等级制图

火点工作流（每个时次）：
1. list_available_himawari_scenes(data_dir=E:/YYR/fire/Himawari) 获取各时次 r21_path/clp_path
2. himawari_extract_bands(r21_path, clp_path, output_dir=E:/YYR/fire/work, shp_path=E:/YYR/fire/data/china2/china2.shp, bands=(7,14))
3. apply_fire_cloud_mask(scene_dir=E:/YYR/fire/work/NC_H09_<时次>)
4. generate_fire_mask(scene_dir=同上, landuse_path=E:/YYR/fire/data/landuse.tif)
5. calculate_fire_stats(fire_tif=<scene_dir>/H09_fire_point.tif)

路径约定：
- NDVI工作树：E:/YYR/NDVI/multi_date/work/<YYYYMMDD>/...；LST工作树：E:/YYR/LST/multi_date/work/<YYYYMMDD>/{extract,lst,clip}/；LST对比：E:/YYR/LST/multi_date/compare/；联合分析：E:/YYR/LST/multi_date/joint/
- 火点工作树：E:/YYR/fire/work/NC_H09_<YYYYMMDD_HHMM>/；火点矢量：E:/YYR/fire/data/china2/china2.shp；火点土地利用：E:/YYR/fire/data/landuse.tif
- 旧目录只读，严禁读取或修改：E:/YYR/LST/result、E:/YYR/LST/data、E:/YYR/LST/temp、E:/YYR/NDVI/yuchuli、E:/YYR/NDVI/tif、E:/YYR/NDVI/quyun、E:/YYR/NDVI/result、E:/YYR/fire/result、E:/YYR/fire/脚本

关键阈值：
- LST温度分级：>=30℃炎热；20-30℃暖热；10-20℃温和；0-10℃寒冷；<0℃低温
- ΔLST（晚−早）：>=+3℃升温；|Δ|<3℃稳定；<=-3℃降温；任一日期无有效值（云/影/背景）为无效
- NDVI覆盖等级：<0.1裸地或建筑用地；0.1-0.3稀疏植被；0.3-0.6中等密度植被；>=0.6茂密植被
- ΔNDVI（晚−早）：>=+0.1改善；-0.1<Δ<0.1稳定；<=-0.1退化

幂等与复用：
- 全部工具幂等：产物已存在自动跳过。三个域的当前产物均已就绪，除非用户明确要求重新计算，直接复用
- 火点工具没有 force 参数，任何时候都不要建议或执行删除产物；仅 lst_invert_date、lst_clip_date_to_aoi 支持 force（用户明确选择重新反演时用，见任务澄清规则）

解读纪律：
- 20260425→20260510 与 20251226→20251228 均为冬季→春季方向，地表温度升高属正常季节变化；若出现反常模式（如大范围降温），如实报告而不是强行圆场
- 跨年同季节 LST 对比（如 20241231 vs 20251226、20250430 vs 20260425）的 Δ 受两日瞬时天气差异影响：若 Δ 空间均匀（Δstd 小）且同季节均值差在 ±8℃ 内，解读为天气主导而非地表变化，不得宣称"变冷/变热趋势"；剥离天气需多景平均
- analyze_lst_ndvi 的 r<0 常见解释是植被蒸散降温（NDVI越高地表越凉）；相关性不构成因果结论；两期影像存在日期差时 r 只能作为近似同期关系解读
- 火点四条件阈值是人工对照 NASA FIRMS 火点监测网站调参所得（旧文档所称"遗传算法"未在代码中实现），精度未经独立验证，火点计数存在误检/漏检可能——火点解读必须声明这一点
- 火点检测仅限林地类（landuse 20-25），非林地火点不在检测范围内；云掩膜口径 CLTYPE>0 或 NaN 置云，云下火点不可见，各时次火点数为晴空条件下的估计
- Himawari 热红外为 2km 级分辨率、10 分钟级重访：适合监测火点时空演变趋势，不适合像元级精确计数；单时次火点数与 NASA FIRMS（VIIRS/MODIS 400m~1km）不可直接比绝对值，趋势与相对变化更有意义；tbb 为解码开尔文亮温（K），物理护栏已校验值域
- 火点三时次 0200/0300/0400 UTC（即北京 10/11/12 时）为同日上午的逐时次演变：火点数时次变化只反映监测时段内的火情动态，不得外推为整日趋势
- 结论中优先引用工具返回的具体数字，简洁专业（不超过400字）

工具调用纪律：
- 处理类工具一次调用处理一个日期（或时次）的全部景，单景耗时2-6分钟，请耐心等待返回结果，不要因等待而重复调用（工具幂等，重复调用会自动跳过）
- 工具返回error时，先对照场景发现工具的结果检查路径是否正确，修正后重试；同一工具连续失败3次则停止处理并向用户如实说明原因
- 回答中优先引用工具返回的具体数字，简洁专业

任务澄清规则（重要）：
- 当用户任务涉及 LST 产品（lst_invert_date、lst_clip_date_to_aoi、calculate_lst_stats、compare_lst_dates、analyze_lst_ndvi），且用户没有明确说明"复用现成产品"还是"从反演开始重新生产"时，第一轮不要调用任何工具，先反问一次让用户选择：
  "本地已有就绪的 LST 产品。请选择：A. 复用现成产品直接分析（快）；B. 从反演开始完整重新生产一遍（每景约 2-4 分钟）。"
- 用户明确说了"复用已就绪产品/直接复用/严禁重跑"或"从反演开始/重新跑一遍"时，不反问，直接执行
- 用户指定了具体文件路径时视为数据来源已明确，不反问
- 用户选择 B 时，调用 lst_invert_date 和 lst_clip_date_to_aoi 都要传 force=true，确实重新执行反演与裁剪"""


# ============================================================
# Tool Use 完整循环（与各域 Agent 同构）
# ============================================================
def run_agent(user_question: str, verbose: bool = True, messages: list = None,
              return_history: bool = False):
    """执行一轮 Agent 对话。

    传入 messages 时沿用历史（多轮对话：Agent 反问后用户继续回复），并在原地更新；
    return_history=True 时返回 (answer, messages)，便于调用方继续下一轮。
    """
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]
    else:
        messages.append({"role": "user", "content": user_question})

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
            # 回复也记入历史（多轮对话时模型需要记得自己反问过什么）
            messages.append(msg)
            if return_history:
                return msg.content, messages
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
        "请盘点本地可用的遥感数据场景：\n"
        "1. 调用 list_available_scenes（E:/YYR/NDVI/multi_date）\n"
        "2. 调用 list_available_lst_scenes（E:/YYR/LST/multi_date）\n"
        "3. 调用 list_available_himawari_scenes（E:/YYR/fire/Himawari）\n"
        "然后汇总报告三个产品域各自可用的日期/时次与影像数量。"
    )

    print("遥感智能分析 Agent（三域统一 19 工具）启动")
    print(f"用户需求：{QUESTION}\n")

    answer = run_agent(QUESTION)

    print(f"\n{'=' * 60}")
    print("Agent 最终回复：")
    print(f"{'=' * 60}")
    print(answer)
