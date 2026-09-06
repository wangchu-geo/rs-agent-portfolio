"""
用问题文件驱动 NDVI+LST 双工具 Agent（镜像 ask_agent.py，换 import 与打印头）。

运行方式：
    PYTHONUTF8=1 python ask_rs_agent.py question.txt
    （question.txt 为 UTF-8 文本，第一行起即完整问题内容）
"""
import sys

from rs_agent import run_agent

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法：python ask_rs_agent.py <问题文件.txt>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        question = f.read().strip()
    if not question:
        print("问题文件为空")
        sys.exit(1)

    print("遥感智能分析 Agent（NDVI+LST 双工具）启动")
    print(f"用户需求：{question}\n")

    answer = run_agent(question)

    print(f"\n{'=' * 60}")
    print("Agent 最终回复：")
    print(f"{'=' * 60}")
    print(answer)
