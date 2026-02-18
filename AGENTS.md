# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

这个仓库是一个 LangChain V1.x 的学习/演示工作区，按章节目录组织（`01_Quickstart` 到 `11_AgentAPIServer`）。

当前代码状态：
- `01_Quickstart` 包含可直接运行的 Python 源码。
- 其余大部分章节目录目前主要是教程型 `README.md`（设计/使用说明），尚未提交对应的 Python 实现。

## 环境与依赖管理

使用 Python 3.11+，并且**统一使用 `uv` 执行依赖安装与脚本运行**（锁文件已提交为 `uv.lock`）。

常用初始化命令：

```bash
uv sync
```

## 运行命令

主要可运行示例：

```bash
uv run python 01_Quickstart/agent.py
```

## 执行约束（必须遵守）

- 执行命令时默认使用 `uv run ...`，不要直接使用 `python ...`。
- 安装依赖时使用 `uv sync`，不要改为 pip 手动安装。
- 只有在用户明确要求或仓库后续文档更新时，才调整上述执行方式。

## 测试与 Lint 现状

仓库根目录当前未配置完整测试套件或 linter 配置（未发现 `tests/`、`pytest` 配置、`ruff`、`mypy` 等）。

在当前仓库状态下，如果被要求“运行测试”，可视为运行演示脚本，主要是：

```bash
uv run python 01_Quickstart/agent.py
```

## 高层架构

### 1) 章节式目录结构

- 根目录 `README.md` 描述了完整的 LangChain / LangGraph / LangSmith 教程路线。
- 每个 `0x_*` 目录对应一个主题（PromptTemplate、Streaming、Memory、RAG、MCP 等）。
- 在当前提交状态中，真正与代码修改直接相关的架构主要集中在 `01_Quickstart`。

### 2) `01_Quickstart` 运行流程

入口：`01_Quickstart/agent.py`

运行管线：
1. 读取配置并初始化日志。
2. 通过 `utils/llms.py` 创建 chat + embedding 模型实例。
3. 通过 `utils/tools.py` 注册工具集合。
4. 使用以下参数创建 LangChain agent：
   - `create_agent(...)`
   - `context_schema=Context`
   - `response_format=ToolStrategy(ResponseFormat)`
   - `checkpointer=InMemorySaver()`
5. 携带 `thread_id` 与 `Context(user_id=...)` 调用 agent。

### 3) 模块职责（`01_Quickstart/utils`）

- `config.py`：集中常量配置（日志路径、日志轮转、LLM 提供商选择）。
- `llms.py`：提供商抽象与模型工厂（`openai` / `oneapi` / `qwen` / `ollama`）。
- `tools.py`：agent 可调用工具定义（天气与用户位置示例）。
- `models.py`：运行时上下文与结构化输出的数据模型。
- `logger.py`：基于 `ConcurrentRotatingFileHandler` 的单例日志管理器。

### 4) 可观测性行为

`01_Quickstart/agent.py` 会根据密钥/证书可用性自动设置 `LANGCHAIN_TRACING_V2`。调试与追踪行为会受以下环境变量影响：
- `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY`
- `SSL_CERT_FILE`、`REQUESTS_CA_BUNDLE`、`CURL_CA_BUNDLE`

## 后续改动配置说明

- `utils/llms.py` 中，`oneapi` 与 `qwen` 依赖环境变量提供凭据。
- 默认提供商由 `01_Quickstart/utils/config.py` 里的 `Config.LLM_TYPE` 控制。
- `Context.user_id` 会影响工具行为（`get_user_location`），重构调用链时应保留。
- 结构化输出契约由 `ResponseFormat` 定义，需与 agent 的 `response_format` 保持一致。

## 当前仓库实际情况（规划时重要）

不要假设每个章节 README 都对应可执行代码。做跨章节实现前，请先确认目标文件是否真实存在。