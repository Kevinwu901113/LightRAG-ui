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
@app.on_event("startup")
async def startup_event():
    global rag

    rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,
            llm_model_name="qwen2.5",
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
            entities_text = knowledge.split("-----实体-----")[1].split("-----关系-----")[0].strip()
            if entities_text:
                # 解析CSV格式的entities数据
                import io
                import pandas as pd
                import csv
                from utils.csv_utils import clean_markdown_csv, parse_csv_with_pipe_quotechar,process_irregular_line_breaks,has_irregular_line_breaks
                
                
               
                csv_text_cleaned = clean_markdown_csv(entities_text)
                final_relationships_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
                
                # 尝试不同的分隔符
                separators = ['+', ',', '\t', '|']
                relationships_data = []
                
                for sep in separators:
                    try:
                        df = pd.read_csv(
                            io.StringIO(final_relationships_text),
                            sep=sep,
                            quotechar=None,
                            quoting=csv.QUOTE_NONE,
                            engine="python",
                            on_bad_lines='skip'  # 跳过有问题的行
                        )
                        if not df.empty:
                            relationships_data = df.to_dict('records')
                            break
                    except Exception:
                        continue
               
                
                # 渲染模板
                html_content = template.render(entities=relationships_data)
                return HTMLResponse(content=html_content)
        except Exception as e:
            print(f"Error parsing entities: {e}")
    
    # 处理relationships数据
    elif datatype == "relationships":
        try:
            relationships_text = knowledge.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
            if relationships_text:
                # 解析CSV格式的relationships数据
                import io
                import pandas as pd
                import csv
                from utils.csv_utils import clean_markdown_csv, parse_csv_with_pipe_quotechar
                
                csv_text_cleaned = clean_markdown_csv(relationships_text)
                final_relationships_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
                
                # 尝试不同的分隔符
                separators = ['+', ',', '\t', '|']
                relationships_data = []
                
                for sep in separators:
                    try:
                        df = pd.read_csv(
                            io.StringIO(final_relationships_text),
                            sep=sep,
                            quotechar=None,
                            quoting=csv.QUOTE_NONE,
                            engine="python",
                            on_bad_lines='skip'  # 跳过有问题的行
                        )
                        if not df.empty:
                            relationships_data = df.to_dict('records')
                            break
                    except Exception:
                        continue
                
                # 渲染模板
                html_content = template.render(relationships=relationships_data)
                return HTMLResponse(content=html_content)
        except Exception as e:
            print(f"Error parsing relationships: {e}")
    
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
    source = await rag.aquery(query_text, param=QueryParam(mode="hybrid", only_need_context=True))# 由于aquery只返回一个值，我们需要创建一个空的source列表
    # print("source:", source)
    # print("response:", response)
    # markdown引用
    base_url = f"{settings.website_address}/v1/references"
    response += f"[^1][^2][^3]"
    response += f"\n\n[^1]: [Entity]({base_url}/entities)\n[^2]: [Relationship]({base_url}/relationships)\n[^3]: [Chunk]({base_url}/chunks)"

    # 保存source数据到当前会话，以便在get_reference中使用
    try:
        import json
        session_file = os.path.join("temp", "chat", "current_session.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            # 更新最新的会话数据
            if len(session_data) > 0:
                latest_session = session_data[0]
                # 保存当前的问题和回答
                latest_session["question"] = query_text
                latest_session["answer"] = response
                
                # 处理knowledge字段，将source分为实体、关系和信息来源三部分
                # 假设source中的数据已经按照一定格式包含了这三部分信息
                # 这里我们需要解析source并构建格式化的数据
                
                # 初始化三部分数据
                # entities_data = []
                # relationships_data = []
                # chunks_data = []
                
                # 解析source数据，使用分隔符"-----实体-----"、"-----关系-----"、"-----信息来源-----"来提取数据
                # if source:
                #     # 确保source是字符串
                #     source_text = source
                #     if isinstance(source, list):
                #         source_text = "\n".join(source)
                    
                    # 使用正则表达式提取各部分内容
                #     import re
                    
                #     # 提取实体部分
                #     entities_match = re.search(r"-----实体-----\s*```csv\s*(.*?)\s*```", source_text, re.DOTALL)
                #     if entities_match:
                #         entities_csv = entities_match.group(1).strip()
                #         # 解析CSV格式的实体数据
                #         lines = entities_csv.split('\n')
                #         if len(lines) > 1:  # 确保有标题行和数据行
                #             # 跳过标题行
                #             for i, line in enumerate(lines[1:], 1):
                #                 if line.strip():  # 确保行不为空
                #                     # 尝试提取ID和内容
                #                     parts = line.split(',', 1)
                #                     if len(parts) >= 2:
                #                         entity_id = parts[0].strip()
                #                         entity_content = parts[1].strip()
                #                         entities_data.append({"id": entity_id, "content": entity_content})
                #                     else:
                #                         entities_data.append({"id": i, "content": line.strip()})
                    
                #     # 提取关系部分
                #     relationships_match = re.search(r"-----关系-----\s*```csv\s*(.*?)\s*```", source_text, re.DOTALL)
                #     if relationships_match:
                #         relationships_csv = relationships_match.group(1).strip()
                #         # 解析CSV格式的关系数据
                #         lines = relationships_csv.split('\n')
                #         if len(lines) > 1:  # 确保有标题行和数据行
                #             # 跳过标题行
                #             for i, line in enumerate(lines[1:], 1):
                #                 if line.strip():  # 确保行不为空
                #                     # 尝试提取ID和内容
                #                     parts = line.split(',', 1)
                #                     if len(parts) >= 2:
                #                         relation_id = parts[0].strip()
                #                         relation_content = parts[1].strip()
                #                         relationships_data.append({"id": relation_id, "content": relation_content})
                #                     else:
                #                         relationships_data.append({"id": i, "content": line.strip()})
                    
                #     # 提取信息来源部分
                #     chunks_match = re.search(r"-----信息来源-----\s*```csv\s*(.*?)\s*```", source_text, re.DOTALL)
                #     if chunks_match:
                #         chunks_csv = chunks_match.group(1).strip()
                #         # 解析CSV格式的信息来源数据
                #         lines = chunks_csv.split('\n')
                #         if len(lines) > 1:  # 确保有标题行和数据行
                #             # 跳过标题行
                #             for i, line in enumerate(lines[1:], 1):
                #                 if line.strip():  # 确保行不为空
                #                     # 尝试提取ID和内容
                #                     parts = line.split(',', 1)
                #                     if len(parts) >= 2:
                #                         chunk_id = parts[0].strip()
                #                         chunk_content = parts[1].strip()
                #                         chunks_data.append({"id": chunk_id, "content": chunk_content})
                #                     else:
                #                         chunks_data.append({"id": i, "content": line.strip()})
                
                # # 构建knowledge字段
                # knowledge_text = "-----实体-----\nid,content\n"
                # for entity in entities_data:
                #     knowledge_text += f"{entity['id']},\"{entity['content']}\"\n"
                
                # knowledge_text += "\n-----关系-----\nid,content\n"
                # for relation in relationships_data:
                #     knowledge_text += f"{relation['id']},\"{relation['content']}\"\n"
                
                # knowledge_text += "\n-----信息来源-----\nid,content\n"
                # for chunk in chunks_data:
                #     knowledge_text += f"{chunk['id']},\"{chunk['content']}\"\n"
                
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




# # lightrag缺少流式生成方法
# async def handle_stream_response(request: gtypes.ChatCompletionRequest, conversation_history) -> StreamingResponse:
#     async def wrapper_astream_search():
#         token_index = 0
#         chat_id = f"chatcmpl-{uuid.uuid4().hex}"
#         full_response = ""
#         async for token in search.astream_search(request.messages[-1].content, conversation_history):  # 调用原始的生成器
#             if token_index == 0:
#                 token_index += 1
#                 continue
#
#             chunk = ChatCompletionChunk(
#                 id=chat_id,
#                 created=int(time.time()),
#                 model=request.model,
#                 object="chat.completion.chunk",
#                 choices=[
#                     Choice(
#                         index=token_index - 1,
#                         finish_reason=None,
#                         delta=ChoiceDelta(
#                             role="assistant",
#                             content=token
#                         )
#                     )
#                 ]
#             )
#             yield f"data: {chunk.json()}\n\n"
#             token_index += 1
#             full_response += token
#
#         content = ""
#         reference = utils.get_reference(full_response)
#         if reference:
#             content = f"\n{utils.generate_ref_links(reference, request.model)}"
#         finish_reason = 'stop'
#         chunk = ChatCompletionChunk(
#             id=chat_id,
#             created=int(time.time()),
#             model=request.model,
#             object="chat.completion.chunk",
#             choices=[
#                 Choice(
#                     index=token_index,
#                     finish_reason=finish_reason,
#                     delta=ChoiceDelta(
#                         role="assistant",
#                         # content=result.context_data["entities"].head().to_string()
#                         content=content
#                     )
#                 ),
#             ],
#         )
#         yield f"data: {chunk.json()}\n\n"
#         yield f"data: [DONE]\n\n"
#
#     return StreamingResponse(wrapper_astream_search(), media_type="text/event-stream")



@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # history = request.messages[:-1]
        # conversation_history = ConversationHistory.from_list([message.dict() for message in history])

        # if not request.stream:
        #     return await handle_sync_response(request, conversation_history)
        # else:
        #     return await handle_stream_response(request, conversation_history)

        return await handle_sync_response(request)

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
