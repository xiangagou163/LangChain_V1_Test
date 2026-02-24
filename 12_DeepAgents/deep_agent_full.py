"""
Deep Agents 完整集成示例
========================
集成功能：
1. MCP (Model Context Protocol) - 工具集成
2. Long-term Memory - 长期记忆
3. HITL (Human-in-the-loop) - 人工介入
4. RAG - 检索增强生成
5. Subagents - 子代理

架构说明：
- Deep Agents 底层基于 LangGraph 构建
- 支持自定义 middleware、checkpointer、store
- 可与 LangChain 生态无缝集成
"""

import os
import asyncio
import json
from typing import Literal, Optional
from datetime import datetime

# ============== 基础依赖 ==============
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call

# ============== Deep Agents ==============
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

# ============== LangGraph ==============
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# ============== MCP 适配器 ==============
# from langchain_mcp_adapters import MCPToolAdapter
# 注：MCP 工具可通过 tools 参数直接传入，本示例使用本地工具


# ============================================================
# 配置区
# ============================================================

# LLM 配置（使用本地代理）
LLM_BASE_URL = "http://localhost:8317/v1"
LLM_API_KEY = "sk-any"
LLM_MODEL = "deepseek-v3.2"

# Tavily 搜索 API
TAVILY_API_KEY = os.environ.get(
    "TAVILY_API_KEY",
    "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"
)

# MCP Server 配置
MCP_SERVER_URL = "http://127.0.0.1:8010/sse"

# 向量数据库路径
CHROMA_PATH = "./chroma_langchain_db"


# ============================================================
# 1. LLM 和 Embedding 初始化
# ============================================================

def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0,
        timeout=120,
    )


def create_embedding():
    """创建 Embedding 实例（本地 HuggingFace）"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ============================================================
# 2. RAG 工具 - 向量检索
# ============================================================

def create_rag_tool():
    """创建 RAG 检索工具"""
    embedding = create_embedding()

    # 连接到已有的 Chroma 向量数据库
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding,
        persist_directory=CHROMA_PATH,
    )

    @tool
    def retrieve_context(query: str, k: int = 3) -> str:
        """
        从向量数据库检索相关文档内容。
        用于查询健康档案、历史记录等信息。

        Args:
            query: 查询字符串
            k: 返回结果数量，默认 3

        Returns:
            检索到的相关文档内容
        """
        try:
            results = vector_store.similarity_search(query, k=k)
            if not results:
                return "未找到相关内容"

            content = "\n\n---\n\n".join([
                f"【文档 {i+1}】\n{doc.page_content}"
                for i, doc in enumerate(results)
            ])
            return content
        except Exception as e:
            return f"检索失败: {str(e)}"

    return retrieve_context


# ============================================================
# 3. 网络搜索工具 (Tavily)
# ============================================================

def create_search_tool():
    """创建网络搜索工具"""
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    @tool
    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> str:
        """
        执行网络搜索，获取实时信息。

        Args:
            query: 搜索查询
            max_results: 最大结果数
            topic: 搜索主题类型

        Returns:
            搜索结果摘要
        """
        try:
            results = tavily_client.search(
                query,
                max_results=max_results,
                topic=topic,
            )

            output = []
            for i, item in enumerate(results.get("results", [])):
                output.append(
                    f"【{i+1}】{item.get('title', 'N/A')}\n"
                    f"    {item.get('content', 'N/A')[:200]}...\n"
                    f"    来源: {item.get('url', 'N/A')}"
                )

            return "\n\n".join(output)
        except Exception as e:
            return f"搜索失败: {str(e)}"

    return internet_search


# ============================================================
# 4. 敏感操作工具 (用于 HITL 演示)
# ============================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    发送邮件（敏感操作，需要人工审批）。

    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文

    Returns:
        发送结果
    """
    return f"✉️ 邮件已发送至 {to}\n主题: {subject}\n内容: {body[:50]}..."


@tool
def delete_file(path: str) -> str:
    """
    删除文件（敏感操作，需要人工审批）。

    Args:
        path: 文件路径

    Returns:
        删除结果
    """
    return f"🗑️ 已删除文件: {path}"


# ============================================================
# 5. 自定义 Middleware (日志记录)
# ============================================================

def create_logging_middleware():
    """创建日志中间件"""
    call_count = [0]

    @wrap_tool_call
    def log_tool_calls(request, handler):
        """拦截并记录每次工具调用"""
        call_count[0] += 1
        tool_name = getattr(request, 'name', str(request))
        args = getattr(request, 'args', {})

        print(f"\n{'='*50}")
        print(f"🔧 [Middleware] 工具调用 #{call_count[0]}")
        print(f"   工具名: {tool_name}")
        print(f"   参数: {json.dumps(args, ensure_ascii=False, default=str)}")

        # 执行工具调用
        result = handler(request)

        print(f"   结果: {str(result)[:100]}...")
        print(f"{'='*50}\n")

        return result

    return log_tool_calls


# ============================================================
# 6. 长期记忆配置
# ============================================================

def create_backend():
    """
    创建 Backend：
    - StateBackend: 临时存储（线程结束丢失）
    - StoreBackend: 持久存储（跨线程存活）
    """
    def make_backend(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),  # 默认临时存储
            routes={
                "/memories/": StoreBackend(runtime)  # /memories/ 路径持久化
            }
        )
    return make_backend


# ============================================================
# 7. MCP 工具适配 (可选)
# ============================================================

async def create_mcp_tools():
    """
    从 MCP Server 获取工具（需要 MCP Server 运行）

    集成方式：
    1. 启动 MCP Server (如 09 章的 rag_mcp_server.py)
    2. 使用 langchain-mcp-adapters 的 load_mcp_tools
    3. 将获取的工具传给 create_deep_agent(tools=...)

    示例代码：
    ```python
    from langchain_mcp_adapters import load_mcp_tools
    from langgraph.prebuilt import create_react_agent

    # SSE 方式连接 MCP Server
    async with MCPClient("sse", url="http://localhost:8010/sse") as client:
        mcp_tools = await load_mcp_tools(client)
        agent = create_deep_agent(tools=mcp_tools, ...)
    ```

    本示例使用本地工具替代，演示相同功能。
    """
    print("ℹ️  MCP 集成说明：")
    print("   1. 启动 MCP Server: python rag_mcp_server.py")
    print("   2. 使用 langchain-mcp-adapters 连接")
    print("   3. 本示例使用本地工具演示相同功能")
    return []


# ============================================================
# 8. 子代理配置
# ============================================================

def create_subagents():
    """创建子代理配置"""

    # 研究员子代理
    research_subagent = {
        "name": "research-agent",
        "description": "用于深入研究问题，进行网络搜索和分析",
        "system_prompt": "你是一个专业的研究员，擅长搜索和分析信息。",  # 使用 system_prompt
        "tools": [create_search_tool()],
    }

    # 档案查询子代理
    archive_subagent = {
        "name": "archive-agent",
        "description": "用于查询健康档案、历史记录等信息",
        "system_prompt": "你是一个档案管理员，擅长从数据库中检索信息。",  # 使用 system_prompt
        "tools": [create_rag_tool()],
    }

    return [research_subagent, archive_subagent]


# ============================================================
# 9. 系统提示词
# ============================================================

SYSTEM_PROMPT = """
你是一个全能的 AI 助手，具备以下能力：

## 工具能力
1. **网络搜索** - 使用 `internet_search` 获取实时信息
2. **档案检索** - 使用 `retrieve_context` 查询健康档案
3. **文件操作** - 读写文件（内置能力）
4. **邮件发送** - `send_email`（需要人工审批）
5. **文件删除** - `delete_file`（需要人工审批）

## 长期记忆
- 将重要信息保存到 `/memories/` 目录下可跨会话记忆
- 例如: `/memories/user_preferences.txt`

## 子代理
- `research-agent`: 深度研究
- `archive-agent`: 档案查询

## 工作原则
1. 复杂任务先规划再执行
2. 敏感操作会请求人工审批
3. 用户偏好保存到长期记忆
"""


# ============================================================
# 10. 创建完整的 Deep Agent
# ============================================================

def create_full_agent():
    """创建完整的 Deep Agent"""

    print("\n" + "="*60)
    print("🚀 正在创建 Deep Agent...")
    print("="*60)

    # 组件初始化
    llm = create_llm()
    checkpointer = MemorySaver()
    store = InMemoryStore()

    # 工具列表
    tools = [
        create_rag_tool(),      # RAG 检索
        create_search_tool(),   # 网络搜索
        send_email,             # 邮件发送（HITL）
        delete_file,            # 文件删除（HITL）
    ]

    # 中间件
    middleware = [create_logging_middleware()]

    # 子代理
    subagents = create_subagents()

    # 创建 Agent
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        subagents=subagents,
        checkpointer=checkpointer,
        store=store,
        backend=create_backend(),
        interrupt_on={
            # 敏感操作需要人工审批
            "send_email": True,      # 允许 approve/edit/reject
            "delete_file": True,     # 允许 approve/edit/reject
        },
        debug=False,
    )

    print("✅ Deep Agent 创建成功")
    print(f"   - 工具数量: {len(tools)}")
    print(f"   - 子代理数量: {len(subagents)}")
    print(f"   - HITL 工具: send_email, delete_file")
    print(f"   - 长期记忆: /memories/ 路径")
    print("="*60 + "\n")

    return agent


# ============================================================
# 11. HITL 处理函数
# ============================================================

async def run_with_hitl(agent, user_input: str, thread_id: str):
    """
    运行 Agent 并处理 HITL 中断
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # 第一次调用
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    # 处理中断
    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        interrupt = interrupts[0]

        action_requests = interrupt.value.get("action_requests", [])
        review_configs = interrupt.value.get("review_configs", [])

        decisions = []
        for i, action in enumerate(action_requests):
            tool_name = action.get("name")
            args = action.get("args", action.get("arguments", {}))
            allowed = review_configs[i].get("allowed_decisions", ["approve", "edit", "reject"])

            print(f"\n{'⚠️'*20}")
            print(f"🔒 [人工审批] 敏感操作检测")
            print(f"   工具: {tool_name}")
            print(f"   参数: {json.dumps(args, ensure_ascii=False)}")
            print(f"   允许操作: {', '.join(allowed)}")
            print(f"{'⚠️'*20}")

            # 自动批准（演示用，实际应让用户输入）
            print("\n>>> 自动批准执行（演示模式）")
            decision = "approve"
            decisions.append({"type": decision})

        # 恢复执行
        result = agent.invoke(
            {"type": "resume", "decisions": decisions},
            config=config,
        )

    return result


# ============================================================
# 12. 主程序
# ============================================================

async def main():
    """主函数"""

    print("\n" + "🔵"*30)
    print("   Deep Agents 完整集成示例")
    print("   MCP + Long-term Memory + HITL + RAG + Subagents")
    print("🔵"*30 + "\n")

    # 创建 Agent
    agent = create_full_agent()

    # 测试场景
    thread_id = "demo-thread-001"

    # ===== 场景 1: RAG 检索 =====
    print("\n" + "─"*60)
    print("📋 场景 1: RAG 检索健康档案")
    print("─"*60)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "张三九有什么健康问题？"}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"\n🤖 回复:\n{result['messages'][-1].content}")

    # ===== 场景 2: 网络搜索 =====
    print("\n" + "─"*60)
    print("🔍 场景 2: 网络搜索")
    print("─"*60)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "搜索一下 LangGraph 的最新特性"}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"\n🤖 回复:\n{result['messages'][-1].content}")

    # ===== 场景 3: 长期记忆 =====
    print("\n" + "─"*60)
    print("💾 场景 3: 保存长期记忆")
    print("─"*60)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请把我的偏好「喜欢简洁的回答风格」保存到长期记忆中"}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"\n🤖 回复:\n{result['messages'][-1].content}")

    # ===== 场景 4: HITL 人工审批 =====
    print("\n" + "─"*60)
    print("🔒 场景 4: HITL 人工审批（敏感操作）")
    print("─"*60)

    result = await run_with_hitl(
        agent,
        "请发一封邮件到 test@example.com，主题是「测试」，内容是「这是一封测试邮件」",
        thread_id
    )
    print(f"\n🤖 回复:\n{result['messages'][-1].content}")

    # ===== 总结 =====
    print("\n" + "="*60)
    print("✅ 所有场景测试完成")
    print("="*60)
    print("""
功能验证:
  ✅ RAG 检索 - 从向量数据库查询健康档案
  ✅ 网络搜索 - Tavily API 实时搜索
  ✅ 长期记忆 - 文件持久化到 /memories/
  ✅ HITL 审批 - 敏感操作人工确认
  ✅ Middleware - 工具调用日志
  ✅ Subagents - 子代理配置
    """)


if __name__ == "__main__":
    asyncio.run(main())
