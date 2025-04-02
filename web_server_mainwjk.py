import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webserver.gtypes import ChatCompletionRequest

from jinja2 import Template
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from webserver import gtypes
from webserver.configs import settings, consts
from query import insert1, query1, prompt_1, prompt_2, prompt_3,autorag,direct_query
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="webserver/static"), name="static")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding, ollama_model_if_cache
from lightrag.utils import EmbeddingFunc
from transformers import AutoTokenizer, AutoModel

rag: LightRAG
WORKING_DIR='./knowledge_base'

# todo:初始化rag逻辑，换成自己的路径
# 在导入部分添加
from lightrag import LightRAG
# 导入流式扩展
import lightrag.stream_extension
from lightrag.utils import EmbeddingFunc
from transformers import AutoTokenizer, AutoModel

rag: LightRAG
WORKING_DIR='./knowledge_base'

# todo:初始化rag逻辑，换成自己的路径
@app.on_event("startup")
async def startup_event():
    global rag

    rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,
            llm_model_name="qwen2.5:7b-instruct-fp16",
            llm_model_max_async=4,
            llm_model_max_token_size=32768,
            llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: ollama_embedding(
                    texts, embed_model="bge-m3", host="http://localhost:11434"
                ),
            ),
        )


@app.get("/")
async def index():
    html_file_path = os.path.join("webserver", "templates", "index.html")
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)



@app.get("/v1/references/{datatype}", response_class=HTMLResponse)
async def get_reference(datatype: str, id: int = None):
    # if not os.path.exists(settings.data):
    #     raise HTTPException(status_code=404, detail=f"Not found")
    if datatype not in ["entities", "claims", "sources", "reports", "relationships", "chunks"]:
        raise HTTPException(status_code=404, detail=f"{datatype} not found")

    html_file_path = os.path.join("webserver", "templates", f"{datatype}_template.html")

    with open(html_file_path, 'r') as file:
        html_content = file.read()
    template = Template(html_content)
    
    # 从会话中获取knowledge数据
    session_data = []
    session_file = os.path.join("temp", "chat", "current_session.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r') as f:
                import json
                session_data = json.load(f)
        except Exception as e:
            print(f"Error loading session data: {e}")
    
    # 从列表中获取最新的会话数据
    knowledge = ""
    if session_data and isinstance(session_data, list) and len(session_data) > 0:
        knowledge = session_data[0].get("knowledge", "")
    
    # 处理entities数据
    if datatype == "entities":
        try:
            import io
            import pandas as pd
            import csv
            import re
            from utils.csv_utils import clean_markdown_csv, parse_csv_with_pipe_quotechar, has_irregular_line_breaks, process_irregular_line_breaks
            
            entities_text = knowledge.split("-----实体-----")[1].split("-----关系-----")[0]
            # 添加预处理步骤
            entities_text = entities_text.strip()
            if not entities_text:
                return HTMLResponse(content="没有找到实体数据")
            
            # 检查是否是CSV格式
            if not any(c in entities_text for c in [',', '\t', '|', '+']):
                return HTMLResponse(content=entities_text)
            
            csv_text_cleaned = clean_markdown_csv(entities_text)
            
            # 检查并处理不规则换行
            if has_irregular_line_breaks(csv_text_cleaned):
                csv_text_cleaned = process_irregular_line_breaks(csv_text_cleaned)
            
            final_entities_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
            
            # 预处理 + 分隔符的特殊情况
            processed_text = final_entities_text
            # 处理引号问题 - 将不成对的引号替换
            processed_text = re.sub(r'(?<!")"(?!")', '\'', processed_text)
            # 确保 + 符号在字段内容中不会被误解为分隔符
            processed_text = re.sub(r'"\+(?=[^"]*")', '"\\+', processed_text)
            
            # 尝试不同的分隔符
            separators = ["+", ',', '\t', '|']
            df = None
            entities_data = []  # 存储实体数据的列表
                
            for sep in separators:
                try:
                    if sep == "+":
                        # 使用更复杂的处理方法来处理 + 分隔符
                        lines = processed_text.split('\n')
                        rows = []
                        
                        for line in lines:
                            if line.strip():
                                # 预处理：保护 {} 内的内容，防止被分割
                                # 先替换 {} 内的 + 号为临时标记
                                protected_line = line
                                # 找出所有 {} 括号内的内容
                                brace_contents = re.findall(r'\{[^{}]*\}', line)
                                
                                # 为每个 {} 内容创建一个临时替代标记
                                for i, content in enumerate(brace_contents):
                                    # 替换 {} 内的 + 为临时标记
                                    safe_content = content.replace('+', '§§PLUS§§')
                                    # 在原始行中替换 {} 内容
                                    protected_line = protected_line.replace(content, safe_content, 1)
                                
                                # 改进：使用正则表达式匹配实体数据的模式
                                # 假设格式为：ID + 实体名 + 类型 + 描述 + 排名
                                # 使用正则表达式匹配前三个字段，然后处理剩余部分
                                match = re.match(r'^(\d+)\s*\+\s*[\'\"]?([^\+]+?)[\'\"]?\s*\+\s*[\'\"]?([^\+]+?)[\'\"]?\s*\+\s*', protected_line)
                                
                                if match:
                                    entity_id = match.group(1).strip()
                                    entity_name = match.group(2).strip()
                                    entity_type = match.group(3).strip()
                                    
                                    # 剩余部分作为描述，直到最后一个+
                                    remaining = protected_line[match.end():]
                                    
                                    # 查找最后一个+号位置
                                    last_plus_pos = remaining.rfind('+')
                                    
                                    if last_plus_pos != -1:
                                        description = remaining[:last_plus_pos].strip()
                                        rank = remaining[last_plus_pos+1:].strip()
                                    else:
                                        # 如果没有找到最后的+，则整个剩余部分作为描述
                                        description = remaining
                                        rank = ""
                                    
                                    # 恢复临时标记为+号
                                    description = description.replace('§§PLUS§§', '+')
                                    
                                    # 移除引号
                                    entity_name = entity_name.strip('\'"')
                                    entity_type = entity_type.strip('\'"')
                                    description = description.strip('\'"')
                                    rank = rank.strip('\'"')
                                    
                                    rows.append({
                                        "id": entity_id,
                                        "entity": entity_name,
                                        "type": entity_type,
                                        "description": description,
                                        "rank": rank
                                    })
                                else:
                                    # 如果不匹配预期格式，尝试简单分割
                                    fields = [f.strip() for f in protected_line.split('+')]
                                    fields = [f.replace('§§PLUS§§', '+') for f in fields]
                                    fields = [f.strip('"\'') for f in fields if f.strip()]
                                    
                                    # if fields and len(fields) >= 5:
                                    #     rows.append({
                                    #         "id": fields[0],
                                    #         "entity": fields[1],
                                    #         "type": fields[2],
                                    #         "description": fields[3],
                                    #         "rank": fields[4]
                                    #     })
                        
                        if rows:
                            entities_data = rows
                            break
                    else:
                        # 对于其他分隔符，使用pandas处理
                        df = pd.read_csv(
                            io.StringIO(processed_text),
                            sep=sep,
                            quotechar='"',
                            escapechar='\\',
                            engine="python",
                            on_bad_lines='skip'  # 跳过有问题的行
                        )
                        
                        if not df.empty:
                            # 确保列名标准化
                            df.columns = [col.strip() for col in df.columns]
                            # 如果列名不是标准的，尝试映射到标准列名
                            if 'id' not in df.columns and df.shape[1] >= 5:
                                df.columns = ['id', 'entity', 'type', 'description', 'rank'][:df.shape[1]]
                            
                            entities_data = df.to_dict('records')
                            break
                except Exception as e:
                    print(f"Error with separator {sep}: {e}")
                    continue
            
            # 检查是否成功解析了数据
            if not entities_data:
                return HTMLResponse(content="无法解析实体数据")
            
            # 渲染模板
            html_content = template.render(entities=entities_data)
            return HTMLResponse(content=html_content)
        except Exception as e:
            print(f"Error parsing entities: {e}")
            return HTMLResponse(content=f"处理实体数据时出错: {str(e)}")
    
    # 处理relationships数据
    elif datatype == "relationships":
        try:
            relationships_text = knowledge.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
            if relationships_text:
                # 解析CSV格式的relationships数据
                import io
                import pandas as pd
                import csv
                from utils.csv_utils import clean_markdown_csv, parse_csv_with_pipe_quotechar, fix_csv_text
                
                # 清理和修复CSV文本
                relationships_text = clean_markdown_csv(relationships_text)
                fixed_csv_text = fix_csv_text(relationships_text)
                relationships_data = []
                
                try:
                    # 尝试使用pandas解析CSV
                    relationships_df = pd.read_csv(io.StringIO(fixed_csv_text), skipinitialspace=True)
                    relationships_df.columns = [col.strip() for col in relationships_df.columns]
                    # 转换为字典列表
                    relationships_data = relationships_df.to_dict('records')
                except Exception as e:
                    # 如果CSV解析失败，尝试手动解析
                    print(f"Error parsing relationships CSV: {e}")
                    lines = relationships_text.strip().split('\n')
                    if len(lines) > 1:
                        header = lines[0].strip().split(',')
                        header = [h.strip() for h in header]
                        for i in range(1, len(lines)):
                            row = lines[i].strip().split(',')
                            if len(row) >= len(header):
                                relation_item = {}
                                for j in range(len(header)):
                                    relation_item[header[j].strip()] = row[j].strip()
                                relationships_data.append(relation_item)
                
                # 处理引号问题，移除内容中的多余引号
                for item in relationships_data:
                    for key in item:
                        if isinstance(item[key], str) and item[key].startswith('"') and item[key].endswith('"'):
                            item[key] = item[key][1:-1]
                
                # 渲染模板
                html_content = template.render(relationships=relationships_data)
                return HTMLResponse(content=html_content)
        except Exception as e:
            print(f"Error parsing relationships: {e}")
            return HTMLResponse(content=f"处理关系数据时出错: {str(e)}")
    
    # 处理chunks/sources数据
    elif datatype == "chunks":
        try:
            if "-----信息来源-----" in knowledge:
                sources_text = knowledge.split("-----信息来源-----")[1].strip()
                if sources_text:
                    # 解析CSV格式的sources数据
                    import io
                    import pandas as pd
                    from utils.csv_utils import clean_markdown_csv, fix_csv_text
                    
                    sources_text = clean_markdown_csv(sources_text)
                    fixed_csv_text = fix_csv_text(sources_text)
                    sources_data = []
                    
                    try:
                        sources_df = pd.read_csv(io.StringIO(fixed_csv_text), skipinitialspace=True)
                        sources_df.columns = [col.strip() for col in sources_df.columns]
                        # 转换为字典列表
                        sources_data = sources_df.to_dict('records')
                    except Exception as e:
                        # 如果CSV解析失败，尝试手动解析
                        print(f"Error parsing sources CSV: {e}")
                        lines = sources_text.strip().split('\n')
                        if len(lines) > 1:
                            header = lines[0].strip().split(',')
                            for i in range(1, len(lines)):
                                row = lines[i].strip().split(',')
                                if len(row) >= len(header):
                                    source_item = {}
                                    for j in range(len(header)):
                                        source_item[header[j].strip()] = row[j].strip()
                                    sources_data.append(source_item)
                    
                    # 处理引号问题，移除内容中的引号
                    for item in sources_data:
                        if 'content' in item and item['content'].startswith('"') and item['content'].endswith('"'):
                            item['content'] = item['content'][1:-1]
                    
                    # 渲染模板
                    html_content = template.render(sources=sources_data)
                    return HTMLResponse(content=html_content)
        except Exception as e:
            print(f"Error parsing chunks: {e}")
    
    # 默认渲染
    html_content = template.render()
    return HTMLResponse(content=html_content)


'''对应html代码（在templates文件夹）
<!DOCTYPE html>
<html>
<head>
    <title>Entity Information</title>
    <style>
        table {
            margin: auto;
            border-collapse: collapse;
            width: 50%;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
            text-align: center;
        }
        td {
            text-align: left;
        }
    </style>
</head>
<body>
    <h1 style="text-align: center;">Entity Information</h1>
    <table>
        <tr><th>Attribute</th><th>Value</th></tr>
        <tr><td>Name</td><td>{{ 111 }}</td></tr>
        <tr><td>Type</td><td>{{ 111 }}</td></tr>
        <tr><td>Description</td><td>{{ 1 }}</td></tr>
{#        <tr><td>Description Embedding</td><td>Length: {{ 1 | length }}</td></tr>#}
        <tr><td>Name Embedding</td><td>{{ 1 }}</td></tr>
        <tr><td>Graph Embedding</td><td>{{ 1 }}</td></tr>
        <tr><td>Community IDs</td><td>{{ 1 }}</td></tr>
        <tr><td>Text Unit IDs</td><td>{{ 1 }}</td></tr>
        <tr><td>Document IDs</td><td>{{ 1 }}</td></tr>
        <tr><td>Rank</td><td>{{ 1 }}</td></tr>
        <tr><td>Attributes</td><td>{{ 1 }}</td></tr>
    </table>
</body>
</html>
'''

async def handle_sync_response(request: ChatCompletionRequest) -> JSONResponse:
    last_msg = request.messages[-1]

    def extract_text(content):
        """将复合 content 转换为纯文本"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return "\n".join(
                block.text for block in content
                if block.type == "text" and block.text.strip()
            )
        return ""  # 处理其他情况

    query_text = extract_text(last_msg.content)

    response = await rag.aquery(query_text, param=QueryParam(mode="hybrid"))
    source = await rag.aquery(query_text, param=QueryParam(mode="hybrid", only_need_context=True))
    
    # markdown引用
    base_url = f"{settings.website_address}/v1/references"
    knowledge_graph_url = f"{settings.website_address}/v1/references/knowledge-graph"
    response += f"[^1][^2][^3]"
    response += f"\n\n[^1]: [Entity]({base_url}/entities)\n[^2]: [Relationship]({base_url}/relationships)\n[^3]: [Chunk]({base_url}/chunks)"

    # 保存source数据到当前会话，以便在get_reference中使用
    try:
        import json
        session_file = os.path.join("temp", "chat", "current_session.json")
        os.makedirs(os.path.dirname(session_file), exist_ok=True)  # 确保目录存在
        
        session_data = []
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        
        # 更新最新的会话数据
        if len(session_data) > 0:
            latest_session = session_data[0]
        else:
            latest_session = {}
            session_data.insert(0, latest_session)
            
        # 保存当前的问题和回答
        latest_session["question"] = query_text
        latest_session["answer"] = response
        latest_session["knowledge"] = source
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
    except Exception as e:
        print(f"Error saving source data: {e}")


    # print("最终回答：", response)

    from openai.types.chat.chat_completion import Choice
    completion = ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=response
                )
            )
        ],
        usage=CompletionUsage(
            completion_tokens=-1,
            prompt_tokens=-1,
            total_tokens=-1
        )
    )
    return JSONResponse(content=jsonable_encoder(completion))

'''refer:
import re
from collections import defaultdict
from typing import Set, Dict

from webserver.configs import settings

pattern = re.compile(r'\[\^Data:(\w+?)\((\d+(?:,\d+)*)\)\]')


def get_reference(text: str) -> dict:
    data_dict = defaultdict(set)
    for match in pattern.finditer(text):
        key = match.group(1).lower()
        value = match.group(2)

        ids = value.replace(" ", "").split(',')
        data_dict[key].update(ids)

    return dict(data_dict)


def generate_ref_links(data: Dict[str, Set[int]], index_id: str) -> str:
    base_url = f"{settings.website_address}/v1/references"
    lines = []
    for key, values in data.items():
        for value in values:
            lines.append(f'[^Data:{key.capitalize()}({value})]: [{key.capitalize()}: {value}]({base_url}/{index_id}/{key}/{value})')
    return "\n".join(lines)
'''




async def handle_stream_response(request: ChatCompletionRequest) -> StreamingResponse:
    async def stream_generator():
        last_msg = request.messages[-1]

        def extract_text(content):
            """将复合 content 转换为纯文本"""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return "\n".join(
                    block.text for block in content
                    if block.type == "text" and block.text.strip()
                )
            return ""  # 处理其他情况

        query_text = extract_text(last_msg.content)
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        token_index = 0
        full_response = ""
        
        # 获取知识库内容（用于后续引用）
        source = await rag.aquery(query_text, param=QueryParam(mode="hybrid", only_need_context=True))
        
        # 保存source数据到当前会话，以便在get_reference中使用
        try:
            import json
            session_file = os.path.join("temp", "chat", "current_session.json")
            os.makedirs(os.path.dirname(session_file), exist_ok=True)  # 确保目录存在
            
            session_data = []
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            
            # 更新最新的会话数据
            if len(session_data) > 0:
                latest_session = session_data[0]
            else:
                latest_session = {}
                session_data.insert(0, latest_session)
                
            # 保存当前的问题和知识库内容
            latest_session["question"] = query_text
            latest_session["knowledge"] = source
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
        except Exception as e:
            print(f"Error saving source data: {e}")
        
        try:
            # 修复参数传递问题
            async for token in rag.astream(
                query_text, 
                param=QueryParam(
                    mode="hybrid",
                    # 移除或修正错误的参数
                    # hashing_kv=some_value  # 如果不需要此参数，直接移除
                )
            ):
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=request.model,
                    object="chat.completion.chunk",
                    choices=[
                        Choice(
                            index=0,
                            finish_reason=None,
                            delta=ChoiceDelta(
                                role="assistant",
                                content=token
                            )
                        )
                    ]
                )
                yield f"data: {chunk.json()}\n\n"
                full_response += token
        except TypeError as e:
            error_msg = f"Error during streaming: {str(e)}"
            print(error_msg)
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",  # 改为合法值
                        delta=ChoiceDelta(
                            role="assistant",
                            content=error_msg
                        )
                    )
                ]
            )
            yield f"data: {chunk.json()}\n\n"
            full_response = error_msg
        except Exception as e:
            error_msg = f"Unexpected error during streaming: {str(e)}"
            print(error_msg)
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",  # 改为合法值
                        delta=ChoiceDelta(
                            role="assistant",
                            content=error_msg
                        )
                    )
                ]
            )
            yield f"data: {chunk.json()}\n\n"
            full_response = error_msg
        
        # 添加引用链接
        base_url = f"{settings.website_address}/v1/references"
        reference_content = f"\n\n[^1]: [Entity]({base_url}/entities)\n[^2]: [Relationship]({base_url}/relationships)\n[^3]: [Chunk]({base_url}/chunks)"
        
        # 发送引用部分
        chunk = ChatCompletionChunk(
            id=chat_id,
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,  # 修复索引号应为0
                    finish_reason=None,
                    delta=ChoiceDelta(
                        role="assistant",
                        content=reference_content
                    )
                )
            ]
        )
        yield f"data: {chunk.json()}\n\n"
        full_response += reference_content
        
        # 更新会话中的回答
        if len(session_data) > 0:
            latest_session["answer"] = full_response
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
        
        # 发送完成标记
        chunk = ChatCompletionChunk(
            id=chat_id,
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,  # 修复索引号
                    finish_reason="stop",
                    delta=ChoiceDelta(
                        role="assistant",
                        content=""
                    )
                )
            ]
        )
        yield f"data: {chunk.json()}\n\n"
        yield f"data: [DONE]\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")



@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # 根据stream参数决定使用哪种响应方式
        if not request.stream:
            return await handle_sync_response(request)
        else:
            return await handle_stream_response(request)

    except Exception as e:
        logger.error(msg=f"chat_completions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 测试请求体
# @app.post("/v1/chat/completions")
# async def chat_completions(request: Request):
#     # 打印请求头
#     headers = dict(request.headers)
#     print("Request Headers:", headers)
#
#     # 读取并打印请求体
#     body_bytes = await request.body()
#     body_str = body_bytes.decode("utf-8")
#     print("Request Body:", body_str)
#
#     # 手动解析请求体（可选）
#     # data = ChatCompletionRequest.parse_raw(body_str)
#     # 后续处理使用data...
#
#     return {"message": "Success"}

'''CherryStudio请求的格式
Request Headers: {'host': 'localhost:20253', 'connection': 'keep-alive', 'content-length': '262', 'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126"', 'x-title': 'Cherry Studio', 'http-referer': 'https://cherry-ai.com', 'x-stainless-retry-count': '0', 'sec-ch-ua-mobile': '?0', 'authorization': 'Bearer', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) CherryStudio/1.1.7 Chrome/126.0.6478.234 Electron/31.7.6 Safari/537.36', 'content-type': 'application/json', 'accept': 'application/json', 'x-api-key': '', 'sec-ch-ua-platform': '"Windows"', 'sec-fetch-site': 'cross-site', 'sec-fetch-mode': 'cors', 'sec-fetch-dest': 'empty', 'accept-encoding': 'gzip, deflate, br, zstd', 'accept-language': 'zh-CN'}
Request Body: {
  "model": "hybrid",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "公明区对外经济办公室有哪些曾用名？"
        }
      ]
    }
  ],
  "temperature": 1,
  "stream": false
}
'''


@app.get("/v1/models", response_model=gtypes.ModelList)
async def list_models():
    models: list[gtypes.Model] = [
        # gtypes.Model(id=consts.INDEX_LOCAL, object="model", created=1644752340, owned_by="graphrag"),
        # gtypes.Model(id=consts.INDEX_GLOBAL, object="model", created=1644752340, owned_by="graphrag"),
        gtypes.Model(id=consts.INDEX_HYBRID, object="model", created=1644752340, owned_by="graphrag")
    ]
    return gtypes.ModelList(data=models)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=settings.server_port)