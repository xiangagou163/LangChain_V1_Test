# LangChain 最新版本 V1.x 中 Agent 的短期记忆

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中实现Agent的短期记忆(对话线程持久化存储)功能，包括短期记忆持久化存储和管理策略、中间件                                                      
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载                  

本期用例的核心功能包含：    

- 短期记忆持久化存储方式一，InMemorySaver，基于内存的短期记忆临时存储         
- 短期记忆持久化存储方式二，PostgresSaver，基于数据库的短期记忆持久化存储       
- 短期记忆管理策略一，修剪消息(Trim Messages)         
- 短期记忆管理策略二，消息摘要(Summarize Messages)   
- 内置中间件、自定义中间件使用

### 短期记忆持久化存储
      
短期记忆允许应用程序在单个对话线程(thread)内记住之前的交互。对话历史是最常见的短期记忆形式               

要为Agent添加短期记忆,需要在创建Agent时指定checkpointer参数              
InMemorySaver 和 PostgresSaver 都是用来做「检查点(checkpointer)」的，实现短期记忆在会话线程里的持久化，但适用场景不同       

#### InMemorySaver

作用：       

把 Agent 的状态（包括消息历史等）保存在进程内存里，进程重启或实例销毁后数据就丢失          

特点：        

- 无需额外依赖，开发环境、本地调试非常方便
- 读写速度快，但不支持跨进程、跨服务共享，也不能在服务重启后恢复线程 

#### PostgresSaver

作用：      
把 Agent 的状态持久化到 PostgreSQL 数据库 中，可以跨进程、跨实例共享，并在重启后恢复对话线程         

特点：         

- 适合生产环境，有可靠持久化和并发访问能力
- 需要维护 Postgres 实例（连接串、权限、备份等），引入一定运维成本

### 短期记忆管理策略

Trim Messages 和 Summarize Messages 都是为了解决“对话太长,超过上下文窗口”的问题,但思路完全不同        

#### 修剪消息(Trim Messages)
      
作用：

在调用 LLM 之前,直接删除对话历史中的部分消息,比如只保留最近几条,让整体消息数量/长度降下来           

特点:

- 实现简单,成本最低(不需要再调一个模型来摘要)
- 被删掉的消息原文和信息都真正丢失,后续模型完全看不到那一段历史
- 适合对“很久以前的细节不重要,最近几轮才关键”的场景,例如闲聊、简单问答

#### 消息摘要(Summarize Messages)

作用：      

当历史消息太长时,用一个聊天模型把早期的对话压缩成摘要,再用这段摘要替换早期原始消息,同时保留最近若干条原文消息             

特点：         

- 更智能,尽量保留早期对话中的关键信息、设定和事实,只是变成浓缩版
- 需要额外模型调用,有一定额外延迟与费用
- 适合需要长期记住用户偏好、背景设定、前文事实的应用,比如复杂助手、长任务协作等 

### 中间件(Middleware)

中间件是预构建的生产级组件，可根据具体需求进行配置，用于处理Agent开发中的常见问题       

#### 内置的中间件
      
LangChain 提供了15种适用于所有 LLM 提供商的中间件          

**(1)对话管理类**      

- 摘要化中间件（Summarization）：当接近Token限制时自动压缩对话历史，保留最近的消息  
- 上下文编辑（Context Editing）：通过清除旧工具调用输出来管理对话上下文，保持上下文窗口可控

**(2)执行控制类**     

- 模型调用限制（Model Call Limit）：限制模型调用次数以防止无限循环或过高成本
- 工具调用限制（Tool Call Limit）：控制工具执行的调用次数，可全局限制或针对特定工具
- 人机协作（Human-in-the-loop）：在工具调用执行前暂停，等待人工审批、编辑或拒绝

**(3)容错与重试类**      

- 模型降级（Model Fallback）：主模型失败时自动切换到备用模型
- 工具重试（Tool Retry）：使用可配置的指数退避自动重试失败的工具调用
- 模型重试（Model Retry）：使用可配置的指数退避自动重试失败的模型调用

**(4)安全与合规类**     

- PII 检测（PII Detection）：检测和处理对话中的个人身份信息，支持编辑、掩码、哈希、阻止等策略

**(5)任务规划类**    

- 待办事项列表（To-do List）：为智能代理配备任务规划和跟踪能力，自动提供 write_todos 工具

**(6)工具优化类**     

- LLM 工具选择器（LLM Tool Selector）：在调用主模型前使用 LLM 智能选择相关工具，适合拥有10+工具的场景
- LLM 工具模拟器（LLM Tool Emulator）：使用 LLM 模拟工具执行以进行测试，无需实际调用外部工具

**(7)开发工具类**      

- Shell 工具（Shell Tool）：向智能代理提供持久的 shell 会话用于命令执行
- 文件搜索（File Search）：提供 Glob 和 Grep 搜索工具用于文件系统检索

**(8)特定供应商中间件**       

此外还提供针对特定 LLM 供应商优化的中间件        

- Anthropic：提供提示缓存、bash 工具、文本编辑器、内存和文件搜索中间件
- OpenAI：提供内容审核中间件

#### 自定义中间件

中间件通过在代理执行流程中的特定节点实现钩子函数来拦截执行，中间件提供两种类型的钩子          

**(1)节点式钩子**      

- 在特定执行点顺序运行,用于日志记录、验证和状态更新
- 包括 before_agent、before_model、after_model、after_agent

**(2)包装式钩子**     

- 围绕每个模型或工具调用运行,用于重试、缓存和转换
- 包括 wrap_model_call 和 wrap_tool_call,你可以决定处理器调用零次(短路)、一次(正常流)或多次(重试逻辑)

**创建方式**      
- 装饰器方式: 快速简单,适合单钩子中间件,使用 @before_model、@after_model 等装饰器包装函数     
- 类方式: 更强大,适合需要多个钩子或配置的复杂中间件,通过继承 AgentMiddleware 类实现   


## 2、准备工作

### 2.1 集成开发环境搭建  

anaconda提供python虚拟环境,pycharm提供集成开发环境          

具体参考如下视频:                         
【大模型应用开发-入门系列】集成开发环境搭建-开发前准备工作                         
https://www.bilibili.com/video/BV1nvdpYCE33/                    
https://youtu.be/KyfGduq5d7w                        

### 2.2 大模型LLM服务接口调用方案

(1)gpt大模型等国外大模型使用方案                   
国内无法直接访问，可以使用Agent的方式，具体Agent方案自己选择                         
这里推荐大家使用:https://nangeai.top/register?aff=Vxlp          

(2)非gpt大模型方案 OneAPI方式或大模型厂商原生接口          

(3)本地开源大模型方案(Ollama方式)            

具体参考如下视频:                                                      
【大模型应用开发-入门系列】大模型LLM服务接口调用方案                   
https://www.bilibili.com/video/BV1BvduYKE75/              
https://youtu.be/mTrgVllUl7Y                           

## 3、项目初始化

关于本期视频的项目初始化请参考本系列的入门案例那期视频，视频链接地址如下:             

【EP01_快速入门用例】2026必学！LangChain最新V1.x版本全家桶LangChain+LangGraph+DeepAgents开发经验免费分享             
https://youtu.be/0ixyKPE2kHQ                    
https://www.bilibili.com/video/BV1EZ62BhEbR/               

### 3.1 下载源码

大家可以在本期视频置顶评论中获取免费资料链接进行下载              
 
### 3.2 构建项目 

使用pycharm构建一个项目，为项目配置虚拟python环境                                            
项目名称：LangChainV1xTest                                                                                     
虚拟环境名称保持与项目名称一致                                                      
 
### 3.3 将相关代码拷贝到项目工程中         

将下载的代码文件夹中的文件全部拷贝到新建的项目根目录下                             

### 3.4 安装项目依赖                

新建命令行终端，在终端中运行如下指令进行安装               
  
```bash
pip install langchain==1.2.1
pip install langchain-openai==1.1.6   
pip install concurrent-log-handler==0.9.28     
pip install langgraph-checkpoint-postgres==3.0.2 
```

**注意:** 建议先使用这里列出的对应版本进行本项目脚本的测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                 

## 4、功能测试   
                                
### 4.1 使用Docker方式运行PostgreSQL数据库和Redis数据库      

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，`cd 04_ShortTermMemory/postgresql` 文件夹下                     
- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                            

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                       

使用数据库客户端软件远程登陆进行可视化操作，这里推荐使用免费的DBeaver客户端软件                     

- DBeaver 客户端软件下载链接: https://dbeaver.io/download/           
            
### 4.2 运行脚本测试            

```bash
python 01_agent_InMemorySaver.py
python 02_agent_PostgresSaver.py 
```


