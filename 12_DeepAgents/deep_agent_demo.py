# Deep Agents 快速入门示例
# Deep Agents 是 LangChain 的高级 Agent 框架，支持自动任务规划、子代理生成等特性

import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI


# 配置 API 密钥
# TAVILY_API_KEY: 用于网络搜索
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")

# 创建 Tavily 搜索客户端
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """
    执行网络搜索

    Args:
        query: 搜索查询字符串
        max_results: 返回结果的最大数量，默认 5
        topic: 搜索主题类型，可选 "general"、"news"、"finance"
        include_raw_content: 是否包含原始网页内容

    Returns:
        搜索结果列表
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# 研究员 Agent 的系统提示词
research_instructions = """你是一位专业的研究员。你的工作是进行全面的研究，然后撰写一份精美的报告。

你可以使用网络搜索工具作为收集信息的主要手段。

## `internet_search`
使用此工具对给定查询进行网络搜索。你可以指定返回结果的最大数量、主题类型，以及是否包含原始内容。

## 工作流程
1. 首先规划研究任务
2. 使用搜索工具收集信息
3. 整合信息并撰写报告
4. 确保报告结构清晰、内容准确
"""


def main():
    """
    主函数：创建并运行 Deep Agent
    """
    # 创建自定义 LLM 模型（使用本地代理）
    # Deep Agents 默认使用 Claude，这里改用本地代理支持的模型
    llm = ChatOpenAI(
        base_url="http://localhost:8317/v1",
        api_key="sk-any",
        model="deepseek-v3.2",
        temperature=0,
        timeout=120,
    )

    # 创建 Deep Agent
    # Deep Agent 内置了以下能力：
    # - write_todos: 自动任务规划
    # - write_file / read_file: 文件管理
    # - create_subagent: 子代理生成
    print("正在创建 Deep Agent...")
    agent = create_deep_agent(
        model=llm,
        tools=[internet_search],
        system_prompt=research_instructions
    )

    # 运行 Agent 进行研究
    question = "什么是 LangGraph？它有什么主要特性？"
    print(f"\n问题: {question}")
    print("-" * 50)

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # 输出结果
    print("\n" + "=" * 50)
    print("Agent 回复:")
    print("=" * 50)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
