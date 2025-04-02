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
async def get_reference(datatype: str):
    # if not os.path.exists(settings.data):
    #     raise HTTPException(status_code=404, detail=f"Not found")
    # if datatype not in ["entities", "claims", "sources", "reports", "relationships"]:
    #     raise HTTPException(status_code=404, detail=f"{datatype} not found")

    html_file_path = os.path.join("webserver", "templates", f"{datatype}_template.html")

    with open(html_file_path, 'r') as file:
        html_content = file.read()
    template = Template(html_content)
    html_content = template.render()
    # html_content = template.render(data=data)
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
    # print("回答：", response)
    # print("来源：", source)
    
    # markdown引用
    base_url = f"{settings.website_address}/v1/references"
    response += f"[^1][^2][^3]"
    response += f"\n\n[^1]: [Entity]({base_url}/entities)\n[^2]: [Relationship]({base_url}/relationships)\n[^3]: [Chunk]({base_url}/chunks)"

    # todo:解析source输出，在get_reference中实现html界面展示。

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



# @app.post("/v1/chat/completions")
# async def chat_completions(request: Request):
#     # 打印请求头
#     headers = dict(request.headers)
#     print("Request Headers:", headers)

#     # 读取并打印请求体
#     body_bytes = await request.body()
#     body_str = body_bytes.decode("utf-8")
#     print("Request Body:", body_str)

#     # 手动解析请求体（可选）
#     # data = ChatCompletionRequest.parse_raw(body_str)
#     # 后续处理使用data...

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
