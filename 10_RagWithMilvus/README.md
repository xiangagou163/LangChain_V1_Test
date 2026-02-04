# LangChain 最新版本 V1.x 中 RAG(检索增强生成，向量数据库选择使用Milvus)    

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中实现RAG(检索增强生成，向量数据库选择使用Milvus)                                                                            
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载                     

本期用例的核心功能包含：    

- RAG知识库构建：创建数据库->定义Schema->配置索引->创建集合->批量插入数据(标量、稀疏向量、稠密向量)
- 知识库搜索：查询、语义搜索、全文搜索、混合搜索
- 构建知识库搜索 MCP Server 对外提供服务 
- Agentic RAG 检索

### 1.1 背景

以看似简单的AI内容搜索为例，用户的需求远非“搜索下关于AI智能体的文章”这样的单维问题，靠着“知识库+向量+LLM”可以轻松解决           
而实际情况可能是:              

- 搜索关于AI智能体相关的文章，文章发布时间在2025.09.05之前，发布者是新智元，返回3篇文章并给出文章的标题、链接、发布者   

这类请求融合了向量相似(文本语义)、关键词全文搜索、条件精确过滤、混合搜索等。问题的瓶颈不在LLM，更不是向量数据库。而是:               

企业AI应用的大量场景从来都不是孤立的，更多的是与企业的业务紧密联系，这体现在应用、流程与数据多个层面                     

本期视频为大家提供一种解决方案思路，以一个完整的闭环应用案例为大家介绍如何实现如下类似搜索:           

- 搜索关于AI智能体相关的文章，文章发布时间在2025.09.05之前，发布者是新智元，返回3篇文章并给出文章的标题、链接、发布者             
- 更甚者，存在无关条件信息的干扰，如:搜索关于AI智能体相关的文章，文章发布时间在2025.09.05之前，发布者是新智元，价格不超过500元，返回3篇文章并给出文章的标题、链接、发布者           

### 1.2 Milvus介绍

Milvus 是一个开源云原生向量数据库，专为在海量向量数据集上进行高性能相似性搜索而设计            
它建立在流行的向量搜索库（包括 Faiss、HNSW、DiskANN 和 SCANN）之上，可为人工智能应用和非结构化数据检索场景提供支持                

官方地址:https://milvus.io/docs/zh                
GitHub地址(42.6k star):https://github.com/milvus-io/milvus                        


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
pip install langchain-text-splitters==1.1.0 
pip install langchain-community==0.4.1
pip install langchain-chroma==1.1.0
pip install pypdf==6.6.0
pip install mcp==1.25.0  
pip install langchain-mcp-adapters==0.2.1
pip install pymilvus==2.6.6

```

**注意:** 建议先使用这里列出的对应版本进行本项目脚本的测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                 


## 4、功能测试   
                                
### 4.1 使用Docker方式运行PostgreSQL数据库和Milvus向量数据库

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，运行如下指令进行部署                     

- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                             
- 进入到 milvus 下执行 `docker-compose up -d` 运行 Milvus 服务                  

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                           

PostgreSQL数据库可使用数据库客户端软件远程登陆进行可视化操作，这里推荐使用免费的DBeaver客户端软件              
 
- DBeaver 客户端软件下载链接: https://dbeaver.io/download/           

### 4.2 功能测试          

```bash
# 1、Milvus向量数据库测试
cd milvus
python 01_create_database.py
python 02_create_collection.py
python 03_insert_data.py
python 04_basic_earch.py
python 05_full_text_search.py
python 06_hybrid_search.py

# 2、MCP Server测试
cd rag_mcp
python mix_text_search.py
python mcp_start.py
python rag_mcp_server_test.py

# 3、Agent测试
python agent_rag.py
```

按照如下参考问题进行测试:           
(1)不指定搜索类型和条件过滤                   
搜索关于多模态大模型持续学习系列研究相关的文章，并给出文章的标题、链接、发布者               

(2)指定搜索类型和条件过滤     
全文搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.05之前，返回3篇文章并给出文章的标题、链接、发布者               
语义搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.05之前，返回3篇文章并给出文章的标题、链接、发布者                
混合搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.05之前，返回3篇文章并给出文章的标题、链接、发布者           

(3)不指定搜索类型但指定条件过滤(存在多个过滤条件)             
搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.05之前，发布者是新智元，返回3篇文章并给出文章的标题、链接、发布者        

(4)不指定搜索类型但指定条件过滤(存在无关过滤条件干扰)            
搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.05之前，发布者是新智元，字数在200字内，价格不超过500元，返回3篇文章并给出文章的标题、链接、发布者        
