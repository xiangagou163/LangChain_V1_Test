# 导入MCP服务器的FastMCP实现
from mcp.server.fastmcp import FastMCP
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 从 langchain_chroma 包中导入 Chroma，用于构建和使用基于 Chroma 的向量数据库/向量存储
from langchain_chroma import Chroma
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 导入日志管理器模块
from utils.logger import LoggerManager



# Author:@南哥AGI研习社 (B站 or YouTube 搜索"南哥AGI研习社")


# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 创建一个名为"rag_mcp_server"的FastMCP服务器实例
mcp = FastMCP(
    name="rag_mcp_server",
    host=Config.MCP_SERVER_HOST,
    port=Config.MCP_SERVER_PORT,
)

# 定义工具：根据查询内容在向量数据库中进行相似度搜索
@mcp.tool()
def retrieve_context(query: str) -> str:
    """
    根据查询内容在向量数据库中进行相似度搜索。

    Args:
        query: 执行搜索的内容

    Returns:
        检索到的文档内容
    """
    try:
        # 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=llm_embedding,
            persist_directory="./chroma_langchain_db",
        )

        # 根据查询内容在向量数据库中进行相似度搜索，返回最相关的 2 个文档
        retrieved_docs = vector_store.similarity_search(query, k=2)
        # 将检索到的文档序列化为字符串格式，包含来源和内容信息
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        logger.info(f"检索到的文本块：{serialized[:200]}...")

        return serialized

    except Exception as e:
        # 记录主程序执行异常的错误日志
        logger.error(f"主程序执行异常: {e}")
        return f"检索失败: {str(e)}"


# 主程序入口
if __name__ == "__main__":
    # 启动 MCP 服务器，使用 streamable-http 传输协议
    mcp.run(transport="streamable-http")
