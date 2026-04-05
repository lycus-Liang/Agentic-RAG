# 🌟 1. 绝对的“第一优先级”：在最顶端加载 .env 环境变量
import os
from dotenv import load_dotenv

# 自动寻找项目根目录下的 .env 文件并加载里面的变量
# override=True 确保 .env 里的值会覆盖系统现有的同名变量
load_dotenv(override=True) 

# 打印一下确认是否加载成功 (测试完可以删掉)
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("🔭 LangSmith 观测台已激活！数据将上报至项目:", os.getenv("LANGCHAIN_PROJECT"))

# ========================================================
# 🌟 2. 之后再导入你的业务代码和其他库
# ========================================================
from src.agent.workflow import build_agentic_rag

def main():
    # 编译你的 Agent 状态机
    app = build_agentic_rag()

    # 准备一个刁钻的问题，测试 Agent 的反思与重试
    inputs = {"question": "语文书里关于那种泥腿子专家的事儿是什么？"}

    print("\n🚀 启动 Agent 工作流...\n")
    
    # 运行状态机并流式输出当前进度
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"✅ 节点 [{key}] 执行完毕。")

    print("\n" + "="*50)
    print("🤖 最终回答：")
    # 打印最终生成的结果
    print(value.get("generation", "未能生成答案。"))
    print("="*50)

if __name__ == "__main__":
    main()