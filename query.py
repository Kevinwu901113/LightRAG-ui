from lightrag import LightRAG, QueryParam
import pandas as pd
from lightrag.llm import ollama_model_complete, ollama_embedding, ollama_model_if_cache
from lightrag.utils import EmbeddingFunc
import datetime
from lightrag.llm import openai_complete_if_cache, openai_embedding, ollama_embedding
import numpy as np
from rich.progress import Progress
from openai import OpenAI
WORKING_DIR = "./dickens1"
xinference_url = 'http://127.0.0.1:9997/v1'
prompt_1="""
---角色---

您是回答有关所提供表中数据的问题的有用助手。


---目标---

通过汇总数据和问题相关的所有信息，生成响应用户问题的回复，并合并任何相关的常识。
如果你不知道答案，就说出来。不要编造任何东西。
请勿在未提供支持证据的情况下提供信息。


---问题相关数据信息---

{context_data}

---用户提出的问题---

{query}

你只需要提供对问题的回复，不要再重复引用数据表中的原文。
"""

prompt_2="""
---角色---

您是根据问题总结整理所提供表中数据的有用助手。


---目标---

通过判断所提供问题相关数据信息中的每条数据内容与用户提出问题的关联性，总结出最相关的数据内容
所提供的数据为csv格式，即以行为单位，每行代表一个数据条目。
你需要保证输出的数据条目数量与输出的目标数据条目数量严格相符
---问题相关数据信息---

{context_data}

---用户提出的问题---

{query}

---输出的目标数据条目数量---

你需要从问题相关数据信息整理{k}条与用户提出问题最相关的数据内容，你需要严格保证整理输出数量为{k}条的最相关的数据，每条数据都需要单独为一行展示

你只需要输出整理后的得到的新数据条目，不需要输出原始数据条目，不要对问题作答，也不要做出解释以及输出其他内容。

######################
-范例-
######################
范例 1:

问题相关数据信息：
id,     entity, type,   description,    rank
1,      "北京市"，"地理位置","{{time: "y1949", place: "北京市", event: "北京市举办开国大典", other: ""}}"<SEP>"{{time: "未知", place: "北京市", event: "北京市位于东经115.7°—117.4°，北纬39.4°—41.6°之间", other: ""}}"<SEP>"{{time: "未知", place: "北京市", event: "北京市是中国的首都", other: ""}}",465
2,     "渤海湾","地理位置","{{time: "未知", place: "北京市", event: "渤海湾与北京市毗邻", other: ""}}",1
3,     "天津市","地理位置","{{time: "未知", place: "北京市", event: "天津市与北京市毗邻", other: ""}}",1
4,     "京津冀城市群","公司或组织名称","{{time: "未知", place: "未知", event: "天津市、北京市与河北省组成京津冀城市群", other: ""}}",60
5,      "庄世良","个人姓名","{{time: "y2003m10d28", place: "公明镇", event: "庄世良被任命为公明镇同心灭罪委员会副主任", other: ""}}"<SEP>"{{time: "未知", place: "公明镇", event: "庄世良担任公明台商联谊会会长", other: ""}}",700
用户问题：
北京的地理位置介绍。
目标数据条目数量:
3条
################
输出:
id,     entity, type,   description
1,      "北京市","地理位置","北京市位于东经115.7°—117.4°，北纬39.4°—41.6°之间,北京市是中国的首都"
2,      "渤海湾","地理位置", "渤海湾与北京市毗邻" 
3,      "天津市","地理位置", "天津市与北京市毗邻" 
################
"""
prompt_3="""
---角色---

您是根据问题总结整理所提供表中数据的有用助手。


---目标---

通过判断所提供问题相关数据信息中的每条数据内容与用户提出问题的关联性，总结出最相关的数据内容
所提供的数据为csv格式，即以行为单位，每行代表一个数据条目。

---问题相关数据信息---

{context_data}

---用户提出的问题---

{query}

---目标数据条目数量---

你需要从问题相关数据信息整理{k}条与用户提出问题最相关的数据内容，你需要严格保证整理输出{k}条最相关的数据，每条数据都需要单独为一行展示

你只需要输出整理后的得到的新数据条目，不需要输出原始数据条目，不要对问题作答，也不要做出解释以及输出其他内容。

######################
-范例-
######################
范例 1:
用户问题：
北京的地理位置介绍。
目标数据条目数量:
3条
问题相关数据信息：
id,     source, target, description,    keywords,       weight, rank
1,      "北京市","天津市","天津市与北京市毗邻","位置",5.0,465
2,      "北京市","渤海湾","渤海湾与北京市毗邻","位置",5.0,444
3,      "北京市","京津冀城市群","天津市、北京市与河北省组成京津冀城市群","组织",5.0,300
4,      "北京市","纬度","北京市纬度是北纬39.4°—41.6°之间","位置",5.0,100
5,      "北京市","经度","北京市经度是东经115.7°—117.4°","位置",5.0,100
################
输出:
id,     source, target, description
1,      "北京市","天津市","天津市与北京市毗邻"
2,      "北京市","渤海湾","渤海湾与北京市毗邻" 
3,      "北京市","经纬度", "北京市纬度是北纬39.4°—41.6°之间,经度是东经115.7°—117.4°" 
################
"""

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[],model_name= "deepseek-chat", **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key='sk-3fcc134c514c480184cb1d949d75fdfb',
        base_url='https://api.deepseek.com',
        **kwargs,
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="bge-m3",
        api_key='EMPTY',
        base_url=xinference_url,
    )

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim

rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

rag1 = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

def insert1(working_dir=WORKING_DIR,text_content=[],model_name='deepseek-chat',i=0):
    if i==0:
        rag2 = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
        )
        rag2.insert([t.decode('utf-8') for t in text_content])
    else :
        rag3 = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
        )
        rag3.insert([t.decode('utf-8') for t in text_content])


async def direct_query(query: str, model_name: str, temperature:float) -> str:
    """直接调用ollama进行查询，不使用知识库"""
    try:
        if model_name=="deepseek-chat":
            try:
                # 使用 OpenAI 官方库直接调用 DeepSeek API
                from openai import AsyncOpenAI
                import httpx
                
                # 添加更多调试信息
                print(f"正在调用 DeepSeek API，查询内容: {query[:50]}...")
                
                # 尝试使用httpx直接发送请求以获取更多错误信息
                async with httpx.AsyncClient(timeout=60.0) as client:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer sk-3fcc134c514c480184cb1d949d75fdfb"
                    }
                    payload = {
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": query}],
                        "temperature": temperature
                    }
                    
                    # 先尝试直接请求以检查连接和响应
                    response = await client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    
                    print(f"DeepSeek API 响应状态码: {response.status_code}")
                    print(f"DeepSeek API 响应头: {response.headers}")
                    print(f"DeepSeek API 响应内容: {response.text[:200]}")
                    
                    if response.status_code != 200:
                        return f"DeepSeek API 错误: 状态码 {response.status_code}, 响应: {response.text}"
                    
                    try:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    except Exception as e:
                        return f"解析 DeepSeek API 响应失败: {str(e)}, 响应内容: {response.text[:200]}"
                
            except Exception as e:
                # 捕获并记录详细错误
                import traceback
                error_msg = f"DeepSeek API 调用失败: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                
                # 尝试使用备用模型
                print("尝试使用本地 Ollama 模型作为备用...")
                try:
                    response = await ollama_model_if_cache(
                        model="qwen2.5",  # 使用默认备用模型
                        prompt=query,
                        temperature=temperature,
                        host="http://localhost:11434",
                        timeout=30
                    )
                    return f"[使用备用模型回答] {response}"
                except:
                    return f"Error querying DeepSeek: {str(e)}"
        else:
            response = await ollama_model_if_cache(
                model=model_name,
                prompt=query,
                temperature=temperature,
                host="http://localhost:11434",
                timeout=30
            )
            return response
    except Exception as e:
        import traceback
        error_msg = f"查询失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 打印到控制台
        return f"Error querying Ollama: {str(e)}"

def query1(working_dir=WORKING_DIR, query='', model_name='deepseek-chat', i=0, param=QueryParam(mode="global", only_need_context=False), use_kb=True,temperature=0.8):
    """
    查询函数，支持使用/不使用知识库
    
    Args:
        working_dir: 知识库目录
        query: 查询内容
        model_name: 模型名称
        i: 模型索引
        param: 查询参数
        use_kb: 是否使用知识库
    """
    if not use_kb:
        # 不使用知识库，直接调用ollama
        import asyncio
        return asyncio.run(direct_query(query, model_name,temperature))
    
    # 使用知识库的原有逻辑
    if i == 0:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            llm_model_name=model_name,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embedding(
                    texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                ),
            ),
        )
        answer = rag.query(query, param=param)
        return answer
    else:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=model_name,
            llm_model_max_async=4,
            llm_model_max_token_size=32768,
            llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embedding(
                    texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                ),
            ),
        )
        answer = rag.query(query, param=param)
        return answer

