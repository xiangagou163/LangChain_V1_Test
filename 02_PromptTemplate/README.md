# LangChain 最新版本 V1.x 中如何定义和使用prompt模版      

## 1、案例介绍

本期视频为大家分享的是如何在 LangChain 最新版本 V1.x 中实现将 prompt 以模版文件(txt、md)形式单独进行维护，并在Agent中加载模版使用                 
涉及到的源码、操作说明文档等全部资料都是开源分享给大家的，大家可以在本期视频置顶评论中获取免费资料链接进行下载          

本期用例的核心功能包含：

- 系统 prompt模版文件定义  
- 用户 prompt模版文件定义
- prompt 模版定义变量
- prompt 变量动态传参
- Agent 加载Prompt模版使用


## 2、准备工作

### 2.1 集成开发环境搭建  

anaconda提供python虚拟环境,pycharm提供集成开发环境          

具体参考如下视频:                         
【大模型应用开发-入门系列】集成开发环境搭建-开发前准备工作                         
https://www.bilibili.com/video/BV1nvdpYCE33/                    
https://youtu.be/KyfGduq5d7w                        

### 2.2 大模型LLM服务接口调用方案

(1)gpt大模型等国外大模型使用方案                   
国内无法直接访问，可以使用代理的方式，具体代理方案自己选择                         
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
```

**注意:** 建议先使用这里列出的对应版本进行本项目脚本的测试，避免因版本升级造成的代码不兼容。测试通过后，可进行升级测试                                      

## 4、功能测试   
                                            
- 运行脚本 `python agent.py`                                  

