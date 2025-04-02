import os
import re
import csv
import io
import json
import shutil
from typing import List, Optional, Union, Dict, Any
import networkx as nx 
import matplotlib.pyplot as plt
import textract
import pandas as pd
import streamlit as st
from lightrag import QueryParam
from query import insert1, query1,prompt_1,prompt_2,prompt_3,direct_query
import ollama
import time
import asyncio
import atexit
from pathlib import Path
from datetime import datetime
# ----------------- 常量定义 ----------------- #
WORKING_DIR = "./dickens1"
WORKING_DIR1 = "tem"
UPLOAD_FOLDER = "uploads"
DEFAULT_TOP_K = 10000
DEFAULT_RECALL_COUNT = 5
SESSION_DIR = Path("./temp/chat")  # 修改会话保存路径
SESSION_DIR.mkdir(exist_ok=True, parents=True)
CURRENT_SESSION = SESSION_DIR / "current_session.json"  # 修改当前会话文件路径

# ----------------- 工具函数 ----------------- #
def register_autosave():
    """注册自动保存功能"""
    # 使用atexit模块注册退出时的回调函数
    import atexit
    atexit.register(save_before_exit)

def save_before_exit():
    """退出时执行保存"""
    if "conversation" in st.session_state and len(st.session_state.conversation) > 0:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
            file_path = SESSION_DIR / filename
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)
            
            # 更新当前会话缓存
            with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
                json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"自动保存失败: {str(e)}")


def get_model_names():
    # 使用ollama的API获取模型列表
    try:
        models = ollama.list()
        if models and 'models' in models and isinstance(models['models'], list):
            # 添加调试信息，查看模型数据结构
            print(f"模型数据结构: {models['models'][0] if models['models'] else '空列表'}")
            # 检查每个模型对象的结构，适应可能的不同键名
            model_names = []
            for model in models['models']:
                if 'name' in model:
                    model_names.append(model['name'])
                elif 'model' in model:
                    model_names.append(model['model'])
                else:
                    # 如果找不到预期的键，尝试获取第一个值作为模型名称
                    if model:
                        first_key = next(iter(model))
                        model_names.append(model[first_key])
            return model_names
        return []
    except Exception as e:
        print(f"获取模型列表失败: {str(e)}")
        # 添加更详细的错误信息
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return []

def clean_markdown_csv(text: str) -> str:
    """清理 Markdown 形式的 CSV 代码块"""
    text = text.strip()
    if text.startswith("```csv"):
        text = text[6:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def fix_csv_text(csv_text: str) -> str:
    """修复多行CSV数据格式"""
    lines = csv_text.strip().splitlines()
    if not lines:
        return csv_text

    header = lines[0].strip()
    fixed_rows = []
    current_row = ""

    for line in lines[1:]:
        line = line.strip()
        if re.match(r'^\d+\s*,', line):
            if current_row:
                fixed_rows.append(current_row)
            current_row = line
        else:
            current_row += " " + line

    if current_row:
        fixed_rows.append(current_row)

    return header + "\n" + "\n".join(fixed_rows)

def parse_csv_with_pipe_quotechar(csv_text: str) -> str:
    """处理CSV文本中的分隔符"""
    def transform_line(line: str) -> str:
        in_quotes = False
        brace_level = 0
        result_chars = []
        i = 0
        length = len(line)

        while i < length:
            if not in_quotes and brace_level == 0 and line.startswith("<SEP>", i):
                result_chars.append('+')
                i += len("<SEP>")
                continue

            c = line[i]
            if c == '"':
                in_quotes = not in_quotes
            elif c == '{':
                brace_level += 1
            elif c == '}':
                if brace_level > 0:
                    brace_level -= 1
            elif c == ',' and not in_quotes and brace_level == 0:
                c = '+'
            
            result_chars.append(c)
            i += 1

        return "".join(result_chars)

    lines = csv_text.splitlines(keepends=False)
    return "\n".join(transform_line(ln) for ln in lines)

# ----------------- 状态管理函数 ----------------- #
def init_session_state():
    """初始化会话状态"""
    defaults = {
        'entity_value': DEFAULT_RECALL_COUNT,
        'relation_value': DEFAULT_RECALL_COUNT,
        'doc_value': DEFAULT_RECALL_COUNT,
        'unlimited_entity': True,
        'unlimited_relation': True,
        'unlimited_doc': True,
        'llm_model': 'qwen2.5',  # 添加默认模型
        'kb_params':{'custom_work_folder':WORKING_DIR},
        'show_settings': False,  # 默认不显示设置面板
        'use_knowledge_base': True,  # 默认启用知识库
        'current_session_name': None,  # 当前会话名称
        'sessions': [],  # 历史会话列表
        'query_submitted': False,  # 新增：查询提交标志
        'temperature': 0.8  # 默认温度值
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # 加载历史会话列表
    if 'sessions' not in st.session_state or not st.session_state.sessions:
        st.session_state.sessions = load_session_history()
    
    # 加载当前会话
    if 'conversation' not in st.session_state:
        st.session_state.conversation = load_current_session()

def reset_session_state():
    """重置会话状态"""
    keys_to_delete = [
        'entity_value', 'relation_value', 'doc_value',
        'use_knowledge_base', 'query_mode', 'llm_model',
        'unlimited_entity', 'unlimited_relation', 'unlimited_doc',
        'show_settings', 
    ]
    # 应该添加 'show_settings'
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# ----------------- UI组件函数 ----------------- #
def create_recall_settings(name: str, key_prefix: str, max_value: int) -> float:
    """创建召回设置UI组件"""
    # 初始化 session state 值
    if f'{key_prefix}_value' not in st.session_state:
        st.session_state[f'{key_prefix}_value'] = DEFAULT_RECALL_COUNT
    if f'unlimited_{key_prefix}' not in st.session_state:
        st.session_state[f'unlimited_{key_prefix}'] = True
    
    def update_value():
        st.session_state[f'{key_prefix}_value'] = st.session_state[f'{key_prefix}_input']

    col1, col2 = st.columns([3, 1])
    with col1:
        count = st.slider(
            f"{name}召回数量",
            min_value=1,
            max_value=max_value,
            value=st.session_state[f'{key_prefix}_value'],
            disabled=st.session_state[f'unlimited_{key_prefix}'],
            key=f'{key_prefix}_slider',
            on_change=lambda: setattr(st.session_state, f'{key_prefix}_value', 
                                    st.session_state[f'{key_prefix}_slider'])
        )
    
    with col2:
        st.number_input(
            f"{name}数量输入",
            min_value=1,
            max_value=max_value,
            value=st.session_state[f'{key_prefix}_value'],
            disabled=st.session_state[f'unlimited_{key_prefix}'],
            label_visibility='collapsed',
            key=f'{key_prefix}_input',
            on_change=update_value
        )
    
    # 移除 value 参数，仅使用 key 来引用 session state
    st.checkbox("不限制", key=f'unlimited_{key_prefix}')
    
    return float('inf') if st.session_state[f'unlimited_{key_prefix}'] else count

def create_file_upload_section(custom_kg_folder: str, custom_upload_folder: str):
    """创建文件上传区域"""
    custom_kg_folder = st.text_input("请输入文件保存知识库路径（或留空使用默认路径）:", value=WORKING_DIR1)
    custom_upload_folder = st.text_input("请输入文件保存路径（或留空使用默认路径）:", value=UPLOAD_FOLDER)
    if not os.path.exists(custom_kg_folder):
        try:
            os.makedirs(custom_kg_folder)
            st.info(f"已创建知识库路径：{custom_kg_folder}")
        except Exception as e:
            st.error(f"创建知识库路径失败：{str(e)}")

    uploaded_files = st.file_uploader("选择文件上传（支持多文件）", accept_multiple_files=True)
    
    if uploaded_files:
        st.subheader("上传的文件:")
        process_uploaded_files(uploaded_files, custom_upload_folder, custom_kg_folder)
    else:
        st.write("没有选择任何文件。")

def process_uploaded_files(uploaded_files, upload_folder: str, kg_folder: str):
    """处理上传的文件"""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    for uploaded_file in uploaded_files:
        save_uploaded_file(uploaded_file, upload_folder)

    st.write("文件正在转化为知识库...")
    text_content = extract_text_from_docs(upload_folder)
    
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1
    insert1(
        working_dir=f"./{kg_folder}",
        text_content=text_content,
        model_name=st.session_state.llm_model,
        i=model_index
    )
    st.write("知识库转化完成")

def save_uploaded_file(uploaded_file, upload_folder: str):
    """保存上传的文件"""
    file_name = uploaded_file.name
    file_path = os.path.join(upload_folder, file_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file, buffer)
    st.write(f"文件 '{file_name}' 已保存到 '{upload_folder}' 文件夹")

def extract_text_from_docs(upload_folder: str) -> List[bytes]:
    """从文档中提取文本"""
    text_content = []
    abs_upload_path = os.path.abspath(upload_folder)
    
    for root, _, files in os.walk(abs_upload_path):
        for f in files:
            if f.endswith(".docx"):
                fp = os.path.join(root, f)
                text_content.append(textract.process(fp))
    
    return text_content

# ----------------- 查询处理函数 ----------------- #
def process_query(query: str, use_kb: bool, params: Dict[str, Any]):
    """处理查询请求"""
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1

    if use_kb:
        try:
            process_kb_query(query, model_index, params)
        except json.JSONDecodeError as e:
            handle_kb_error(e, params['custom_work_folder'])
        except Exception as e:
            st.error(f"查询过程中发生错误：{str(e)}")
    else:
        answer=process_direct_query(query, model_index,st.session_state.temperature)
        st.session_state.conversation.append({
            "question": query,
            "answer": answer,
            "knowledge": "模型没有根据知识库做出的直接回答",
            "timestamp": datetime.now().isoformat()
        })
    
    # 保存当前会话
    with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
        json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)


def process_kb_query(query: str, model_index: int, params: Dict[str, Any]):
    """处理基于知识库的查询"""
    work_dirs = params['custom_work_folder'].split(',')
    context1 = ""
    context2 = ""
    context3 = ""
    
    # 验证所有知识库路径
    valid_dirs = []
    for dir in work_dirs:
        dir = dir.strip()  # 去除可能的空格
        if not dir:
            continue
        if not os.path.exists(dir):
            st.warning(f"知识库路径 '{dir}' 不存在！将被跳过。")
            continue
        valid_dirs.append(dir)
    
    if not valid_dirs:
        st.error("没有有效的知识库路径！请先上传文件创建知识库。")
        return
        
    # 继续处理有效的知识库路径
    for dir in valid_dirs:
        # 现有的处理逻辑...

    # 获取模型名称
        model_name = st.session_state.llm_model
    
    # 获取上下文和答案
        context = query1(
            query=query,
            param=QueryParam(
                mode=params['query_mode'],
                only_need_context=True
            ),
            i=model_index,
            model_name=model_name,
            working_dir=dir,
            use_kb=True,  # 确保使用知识库
            temperature=st.session_state.temperature
        )
        
        entities_text = context.split("-----实体-----")[1].split("-----关系-----")[0]
        if(params['entity_count']!=float('inf')):
            
            e_prompt=prompt_2.format(k=int(params['entity_count']),query=query,context_data=entities_text)
           
            context1+=asyncio.run(direct_query(query=e_prompt,model_name=st.session_state.llm_model,temperature=st.session_state.temperature))
            context1+='\n'
            
        else:
            context1+=entities_text
            context1+='\n'
        relation_text=context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
        if(params['relation_count']!=float('inf')):
            
            e_prompt=prompt_3.format(k=int(params['relation_count']),query=query,context_data=relation_text)
            context2+=asyncio.run(direct_query(query=e_prompt,model_name=st.session_state.llm_model,temperature=st.session_state.temperature))
            context2+='\n'
        else:
            context2+=relation_text
            context2+='\n'
        sources_text=context.split("-----信息来源-----")[1].strip()
        if(params['doc_count']!=float('inf')):
            
            sources_text = clean_markdown_csv(sources_text)
            
            fixed_csv_text = fix_csv_text(sources_text)
            
            e_prompt=prompt_2.format(k=int(params['doc_count']),query=query,context_data=fixed_csv_text)
            
            context3+=asyncio.run(direct_query(query=e_prompt,model_name=st.session_state.llm_model,temperature=st.session_state.temperature))
            context3+='\n'
        else:
            context3+=sources_text
            context3+='\n'
    context="-----实体-----\n"+context1+"-----关系-----\n"+context2+"-----信息来源-----\n"+context3    
    a_prompt=prompt_1.format(query=query,context_data=context)
    # 打印prompt用于调试
    # st.write("Debug - Prompt内容:")
    # st.write(a_prompt)
    
    if context:
            st.header("Source Context")
            display_entities(context, params['entity_count'])
            display_relationships(context, params['relation_count'])
            display_sources(context, params['doc_count'])
    
    # 修改模型索引为0（如果是deepseek-chat）或保持为1（其他模型）
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1
    
    # 尝试简化prompt，只保留最关键的信息
    simplified_prompt = f"基于以下信息回答问题：\n\n问题：{query}\n\n相关信息：\n{context}\n\n请详细回答。"
    
    # 使用query1函数调用模型
    answer = query1(
        query=simplified_prompt,  # 使用简化的prompt
        i=model_index,  # 使用正确的模型索引
        model_name=st.session_state.llm_model,
        use_kb=False,
        temperature=st.session_state.temperature
    )
    st.header("Generated Answer")
    st.write(answer)
    
    st.session_state.conversation.append({
            "question": query,
            "answer": answer,
            "knowledge": context,
            "timestamp": datetime.now().isoformat()
        })
    # answer = query1(
    #     query=query,
    #     param=QueryParam(
    #         mode=params['query_mode'],
    #         only_need_context=False
    #     ),
    #     i=model_index,
    #     model_name=model_name,
    #     working_dir=params['custom_work_folder'],
    #     use_kb=True  # 确保使用知识库

    

def process_direct_query(query: str, model_index: int, temperature: float = 0.8):
    """处理直接查询"""
    model_name = st.session_state.llm_model
    answer = query1(
        query=query,
        i=model_index,
        model_name=model_name,
        use_kb=False,
        temperature=temperature
    )
    st.header("Generated Answer")
    st.write(answer)
    return answer

def handle_kb_error(error: json.JSONDecodeError, work_folder: str):
    """处理知识库错误"""
    st.error("知识库文件损坏！请尝试重新上传文件创建知识库。")
    st.error(f"错误详情：{str(error)}")
    try:
        reset_corrupted_kb_files(work_folder)
        st.info("已重置损坏的知识库文件。请重新上传文件创建知识库。")
    except Exception as e:
        st.error(f"重置知识库文件失败：{str(e)}")

def reset_corrupted_kb_files(work_folder: str):
    """重置损坏的知识库文件"""
    json_files = [
        os.path.join(work_folder, f)
        for f in os.listdir(work_folder)
        if f.endswith('.json')
    ]
    for json_file in json_files:
        with open(json_file, 'w') as f:
            json.dump({}, f)

def display_query_results(answer: str, context: Optional[str], params: Dict[str, Any]):
    """显示查询结果"""
    st.header("Generated Answer")
    st.write(answer)

    if context:
        st.header("Source Context")
        st.markdown("### 📊 实体")
        display_entities(context, params['entity_count'])
        st.markdown("### 🔄 关系")
        display_relationships(context, params['relation_count'])
        st.markdown("### 📄 文档")
        display_sources(context, params['doc_count'])

def display_entities(context: str, entity_count: float):
    """显示实体信息"""
    if "-----实体-----" in context:
        # 移除外层expander
        try:
            entities_text = context.split("-----实体-----")[1].split("-----关系-----")[0]
            # 添加预处理步骤
            entities_text = entities_text.strip()
            if not entities_text:
                st.warning("没有找到实体数据")
                return
            
            # 检查是否是CSV格式
            if not any(c in entities_text for c in [',', '\t', '|', '+']):
                st.text(entities_text)
                return
            
            csv_text_cleaned = clean_markdown_csv(entities_text)
            final_entities_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
            
            # 尝试不同的分隔符
            separators = ['+', ',', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.StringIO(final_entities_text),
                        sep=sep,
                        quotechar=None,
                        quoting=csv.QUOTE_NONE,
                        engine="python",
                        on_bad_lines='skip'  # 跳过有问题的行
                    )
                    if not df.empty:
                        break
                except Exception:
                    continue
            
            if df is not None and not df.empty:
                st.dataframe(df)
            else:
                st.text(entities_text)
                
        except Exception as e:
            st.error(f"解析实体数据时出错: {str(e)}")

def display_relationships(context: str, relation_count: float):
    """显示关系信息"""
    if "-----关系-----" in context:
        # 移除外层expander
        try:
            relationships_text = context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
            relationships_df = pd.read_csv(io.StringIO(relationships_text))
            st.dataframe(relationships_df)
        except Exception as e:
            st.text(f"Error parsing relationships: {e}")
            st.text(relationships_text)

def display_sources(context: str, doc_count: float):
    """显示来源信息"""
    if "-----信息来源-----" in context:
        # 移除外层expander
        try:
            sources_text = context.split("-----信息来源-----")[1].strip()
            sources_text = clean_markdown_csv(sources_text)
            fixed_csv_text = fix_csv_text(sources_text)
            sources_df = pd.read_csv(io.StringIO(fixed_csv_text), skipinitialspace=True)
            sources_df.columns = [col.strip() for col in sources_df.columns]

            markdown_content = format_sources_to_markdown(sources_df)
            st.markdown(markdown_content)
        except Exception as e:
            st.text(f"Error parsing sources: {e}")
            st.text(sources_text)

def format_sources_to_markdown(df: pd.DataFrame) -> str:
    """将数据框格式化为Markdown文本"""
    markdown_content = ""
    for _, row in df.iterrows():
        markdown_content += f"### **{row['id']}**\n"
        for col in df.columns:
            if col != 'id':
                markdown_content += f"**{col}:** {row[col]}\n"
            markdown_content += "\n"
    return markdown_content

# ----------------- 会话管理函数 ----------------- #
def load_session_history():
    """加载所有历史会话"""
    sessions = []
    for file in SESSION_DIR.glob("*.json"):
        if file.name == "current_session.json":
            continue
        try:
            # 从文件名中提取时间戳
            timestamp = file.stem.replace("session_", "")
            # 尝试解析时间戳
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                display_time = timestamp
                
            # 读取会话内容获取消息数量
            with open(file, "r", encoding="utf-8") as f:
                conversation = json.load(f)
                message_count = len(conversation)
            
            sessions.append({
                "filename": file.name,
                "display_name": f"{display_time} ({message_count}条消息)",
                "timestamp": timestamp
            })
        except Exception as e:
            print(f"加载会话 {file.name} 失败: {str(e)}")
    
    # 按时间戳排序，最新的在前面
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def load_conversation(filename):
    """加载指定的会话文件"""
    try:
        file_path = SESSION_DIR / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"加载会话失败: {str(e)}")
        return []

def save_conversation(conversation, filename=None):
    """保存会话到文件"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
        
        file_path = SESSION_DIR / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        # 同时更新当前会话
        with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
            
        return filename
    except Exception as e:
        st.error(f"保存会话失败: {str(e)}")
        return None

def delete_conversation(filename):
    """删除指定的会话文件"""
    try:
        file_path = SESSION_DIR / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"删除会话失败: {str(e)}")
        return False

def load_current_session():
    """加载当前会话"""
    if CURRENT_SESSION.exists():
        try:
            with open(CURRENT_SESSION, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def reset_session_state():
    """重置会话状态"""
    keys_to_delete = [
        'entity_value', 'relation_value', 'doc_value',
        'use_knowledge_base', 'query_mode', 'llm_model',
        'unlimited_entity', 'unlimited_relation', 'unlimited_doc',
        'show_settings', 
    ]
    # 应该添加 'show_settings'
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# ----------------- UI组件函数 ----------------- #
def create_recall_settings(name: str, key_prefix: str, max_value: int) -> float:
    """创建召回设置UI组件"""
    # 初始化 session state 值
    if f'{key_prefix}_value' not in st.session_state:
        st.session_state[f'{key_prefix}_value'] = DEFAULT_RECALL_COUNT
    if f'unlimited_{key_prefix}' not in st.session_state:
        st.session_state[f'unlimited_{key_prefix}'] = True
    
    def update_value():
        st.session_state[f'{key_prefix}_value'] = st.session_state[f'{key_prefix}_input']

    col1, col2 = st.columns([3, 1])
    with col1:
        count = st.slider(
            f"{name}召回数量",
            min_value=1,
            max_value=max_value,
            value=st.session_state[f'{key_prefix}_value'],
            disabled=st.session_state[f'unlimited_{key_prefix}'],
            key=f'{key_prefix}_slider',
            on_change=lambda: setattr(st.session_state, f'{key_prefix}_value', 
                                    st.session_state[f'{key_prefix}_slider'])
        )
    
    with col2:
        st.number_input(
            f"{name}数量输入",
            min_value=1,
            max_value=max_value,
            value=st.session_state[f'{key_prefix}_value'],
            disabled=st.session_state[f'unlimited_{key_prefix}'],
            label_visibility='collapsed',
            key=f'{key_prefix}_input',
            on_change=update_value
        )
    
    # 移除 value 参数，仅使用 key 来引用 session state
    st.checkbox("不限制", key=f'unlimited_{key_prefix}')
    
    return float('inf') if st.session_state[f'unlimited_{key_prefix}'] else count

def create_file_upload_section(custom_kg_folder: str, custom_upload_folder: str):
    """创建文件上传区域"""
    custom_kg_folder = st.text_input("请输入文件保存知识库路径（或留空使用默认路径）:", value=WORKING_DIR1)
    custom_upload_folder = st.text_input("请输入文件保存路径（或留空使用默认路径）:", value=UPLOAD_FOLDER)
    if not os.path.exists(custom_kg_folder):
        try:
            os.makedirs(custom_kg_folder)
            st.info(f"已创建知识库路径：{custom_kg_folder}")
        except Exception as e:
            st.error(f"创建知识库路径失败：{str(e)}")

    uploaded_files = st.file_uploader("选择文件上传（支持多文件）", accept_multiple_files=True)
    
    if uploaded_files:
        st.subheader("上传的文件:")
        process_uploaded_files(uploaded_files, custom_upload_folder, custom_kg_folder)
    else:
        st.write("没有选择任何文件。")

def process_uploaded_files(uploaded_files, upload_folder: str, kg_folder: str):
    """处理上传的文件"""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    for uploaded_file in uploaded_files:
        save_uploaded_file(uploaded_file, upload_folder)

    st.write("文件正在转化为知识库...")
    text_content = extract_text_from_docs(upload_folder)
    
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1
    insert1(
        working_dir=f"./{kg_folder}",
        text_content=text_content,
        model_name=st.session_state.llm_model,
        i=model_index
    )
    st.write("知识库转化完成")

def save_uploaded_file(uploaded_file, upload_folder: str):
    """保存上传的文件"""
    file_name = uploaded_file.name
    file_path = os.path.join(upload_folder, file_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file, buffer)
    st.write(f"文件 '{file_name}' 已保存到 '{upload_folder}' 文件夹")

def extract_text_from_docs(upload_folder: str) -> List[bytes]:
    """从文档中提取文本"""
    text_content = []
    abs_upload_path = os.path.abspath(upload_folder)
    
    for root, _, files in os.walk(abs_upload_path):
        for f in files:
            if f.endswith(".docx"):
                fp = os.path.join(root, f)
                text_content.append(textract.process(fp))
    
    return text_content

# ----------------- 查询处理函数 ----------------- #
def process_query(query: str, use_kb: bool, params: Dict[str, Any]):
    """处理查询请求"""
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1

    if use_kb:
        try:
            process_kb_query(query, model_index, params)
        except json.JSONDecodeError as e:
            handle_kb_error(e, params['custom_work_folder'])
        except Exception as e:
            st.error(f"查询过程中发生错误：{str(e)}")
    else:
        answer=process_direct_query(query, model_index,st.session_state.temperature)
        st.session_state.conversation.append({
            "question": query,
            "answer": answer,
            "knowledge": "模型没有根据知识库做出的直接回答",
            "timestamp": datetime.now().isoformat()
        })
    
    # 保存当前会话到current_session.json
    try:
        with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)
        
        # 刷新会话列表
        st.session_state.sessions = load_session_history()
    except Exception as e:
        print(f"保存当前会话失败: {str(e)}")


def process_kb_query(query: str, model_index: int, params: Dict[str, Any]):
    """处理基于知识库的查询"""
    work_dirs = params['custom_work_folder'].split(',')
    context1 = ""
    context2 = ""
    context3 = ""
    
    # 验证所有知识库路径
    valid_dirs = []
    for dir in work_dirs:
        dir = dir.strip()  # 去除可能的空格
        if not dir:
            continue
        if not os.path.exists(dir):
            st.warning(f"知识库路径 '{dir}' 不存在！将被跳过。")
            continue
        valid_dirs.append(dir)
    
    if not valid_dirs:
        st.error("没有有效的知识库路径！请先上传文件创建知识库。")
        return
        
    # 继续处理有效的知识库路径
    for dir in valid_dirs:
        # 现有的处理逻辑...

    # 获取模型名称
        model_name = st.session_state.llm_model
    
    # 获取上下文和答案
        context = query1(
            query=query,
            param=QueryParam(
                mode=params['query_mode'],
                only_need_context=True
            ),
            i=model_index,
            model_name=model_name,
            working_dir=dir,
            use_kb=True,  # 确保使用知识库
            temperature=st.session_state.temperature
        )
        
        entities_text = context.split("-----实体-----")[1].split("-----关系-----")[0]
        if(params['entity_count']!=float('inf')):
            
            e_prompt=prompt_2.format(k=int(params['entity_count']),query=query,context_data=entities_text)
           
            context1+=asyncio.run(direct_query(query=e_prompt,model_name=st.session_state.llm_model,temperature=st.session_state.temperature))
            context1+='\n'
            
        else:
            context1+=entities_text
            context1+='\n'
        relation_text=context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
        if(params['relation_count']!=float('inf')):
            
            e_prompt=prompt_3.format(k=int(params['relation_count']),query=query,context_data=relation_text)
            context2+=asyncio.run(direct_query(query=e_prompt,model_name=st.session_state.llm_model,temperature=st.session_state.temperature))
            context2+='\n'
        else:
            context2+=relation_text
            context2+='\n'
        sources_text=context.split("-----信息来源-----")[1].strip()
        if(params['doc_count']!=float('inf')):
            
            sources_text = clean_markdown_csv(sources_text)
            
            fixed_csv_text = fix_csv_text(sources_text)
            
            e_prompt=prompt_2.format(k=int(params['doc_count']),query=query,context_data=fixed_csv_text)
            
            context3+=asyncio.run(direct_query(query=e_prompt,model_name=st.session_state.llm_model,temperature=st.session_state.temperature))
            context3+='\n'
        else:
            context3+=sources_text
            context3+='\n'
    context="-----实体-----\n"+context1+"-----关系-----\n"+context2+"-----信息来源-----\n"+context3    
    a_prompt=prompt_1.format(query=query,context_data=context)
    # 打印prompt用于调试
    # st.write("Debug - Prompt内容:")
    # st.write(a_prompt)
    
    if context:
            st.header("Source Context")
            display_entities(context, params['entity_count'])
            display_relationships(context, params['relation_count'])
            display_sources(context, params['doc_count'])
    
    # 修改模型索引为0（如果是deepseek-chat）或保持为1（其他模型）
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1
    
    # 尝试简化prompt，只保留最关键的信息
    simplified_prompt = f"基于以下信息回答问题：\n\n问题：{query}\n\n相关信息：\n{context}\n\n请详细回答。"
    
    # 使用query1函数调用模型
    answer = query1(
        query=simplified_prompt,  # 使用简化的prompt
        i=model_index,  # 使用正确的模型索引
        model_name=st.session_state.llm_model,
        use_kb=False,
        temperature=st.session_state.temperature
    )
    st.header("Generated Answer")
    st.write(answer)
    
    st.session_state.conversation.append({
            "question": query,
            "answer": answer,
            "knowledge": context,
            "timestamp": datetime.now().isoformat()
        })
    # answer = query1(
    #     query=query,
    #     param=QueryParam(
    #         mode=params['query_mode'],
    #         only_need_context=False
    #     ),
    #     i=model_index,
    #     model_name=model_name,
    #     working_dir=params['custom_work_folder'],
    #     use_kb=True  # 确保使用知识库

    

def process_direct_query(query: str, model_index: int, temperature: float = 0.8):
    """处理直接查询"""
    model_name = st.session_state.llm_model
    answer = query1(
        query=query,
        i=model_index,
        model_name=model_name,
        use_kb=False,
        temperature=temperature
    )
    st.header("Generated Answer")
    st.write(answer)
    return answer

def handle_kb_error(error: json.JSONDecodeError, work_folder: str):
    """处理知识库错误"""
    st.error("知识库文件损坏！请尝试重新上传文件创建知识库。")
    st.error(f"错误详情：{str(error)}")
    try:
        reset_corrupted_kb_files(work_folder)
        st.info("已重置损坏的知识库文件。请重新上传文件创建知识库。")
    except Exception as e:
        st.error(f"重置知识库文件失败：{str(e)}")

def reset_corrupted_kb_files(work_folder: str):
    """重置损坏的知识库文件"""
    json_files = [
        os.path.join(work_folder, f)
        for f in os.listdir(work_folder)
        if f.endswith('.json')
    ]
    for json_file in json_files:
        with open(json_file, 'w') as f:
            json.dump({}, f)

def display_query_results(answer: str, context: Optional[str], params: Dict[str, Any]):
    """显示查询结果"""
    st.header("Generated Answer")
    st.write(answer)

    if context:
        st.header("Source Context")
        st.markdown("### 📊 实体")
        display_entities(context, params['entity_count'])
        st.markdown("### 🔄 关系")
        display_relationships(context, params['relation_count'])
        st.markdown("### 📄 文档")
        display_sources(context, params['doc_count'])

def display_entities(context: str, entity_count: float):
    """显示实体信息"""
    if "-----实体-----" in context:
        # 移除外层expander
        try:
            entities_text = context.split("-----实体-----")[1].split("-----关系-----")[0]
            # 添加预处理步骤
            entities_text = entities_text.strip()
            if not entities_text:
                st.warning("没有找到实体数据")
                return
            
            # 检查是否是CSV格式
            if not any(c in entities_text for c in [',', '\t', '|', '+']):
                st.text(entities_text)
                return
            
            csv_text_cleaned = clean_markdown_csv(entities_text)
            final_entities_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
            
            # 尝试不同的分隔符
            separators = ['+', ',', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.StringIO(final_entities_text),
                        sep=sep,
                        quotechar=None,
                        quoting=csv.QUOTE_NONE,
                        engine="python",
                        on_bad_lines='skip'  # 跳过有问题的行
                    )
                    if not df.empty:
                        break
                except Exception:
                    continue
            
            if df is not None and not df.empty:
                st.dataframe(df)
            else:
                st.text(entities_text)
                
        except Exception as e:
            st.error(f"解析实体数据时出错: {str(e)}")

def display_relationships(context: str, relation_count: float):
    """显示关系信息"""
    if "-----关系-----" in context:
        # 移除外层expander
        try:
            relationships_text = context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
            relationships_df = pd.read_csv(io.StringIO(relationships_text))
            st.dataframe(relationships_df)
        except Exception as e:
            st.text(f"Error parsing relationships: {e}")
            st.text(relationships_text)

def display_sources(context: str, doc_count: float):
    """显示来源信息"""
    if "-----信息来源-----" in context:
        # 移除外层expander
        try:
            sources_text = context.split("-----信息来源-----")[1].strip()
            sources_text = clean_markdown_csv(sources_text)
            fixed_csv_text = fix_csv_text(sources_text)
            sources_df = pd.read_csv(io.StringIO(fixed_csv_text), skipinitialspace=True)
            sources_df.columns = [col.strip() for col in sources_df.columns]

            markdown_content = format_sources_to_markdown(sources_df)
            st.markdown(markdown_content)
        except Exception as e:
            st.text(f"Error parsing sources: {e}")
            st.text(sources_text)

def format_sources_to_markdown(df: pd.DataFrame) -> str:
    """将数据框格式化为Markdown文本"""
    markdown_content = ""
    for _, row in df.iterrows():
        markdown_content += f"### **{row['id']}**\n"
        for col in df.columns:
            if col != 'id':
                markdown_content += f"**{col}:** {row[col]}\n"
            markdown_content += "\n"
    return markdown_content

# ----------------- 主程序入口 ----------------- #
# ----------------- 主程序入口 ----------------- #
def main():
    """主程序入口"""
    st.set_page_config(page_title="hello",page_icon="👏")
    st.title("LightRAG Query Interface")
    init_session_state()
    register_autosave()  # 注册自动保存
    
    if 'model_names' not in st.session_state:
        s=["deepseek-chat"]
        s.extend(get_model_names())
        st.session_state.model_names = s
    
    # 创建侧边栏
    with st.sidebar:
        # 创建一个容器用于显示主要内容
        main_content = st.container()
        
        with main_content:
            if not st.session_state.show_settings:
                # 显示历史会话区域
                st.markdown("### 历史会话")
                
                # 新建会话按钮
                if st.button("➕ 新建会话", key="new_session"):
                    # 保存当前会话
                    if len(st.session_state.conversation) > 0:
                        save_conversation(st.session_state.conversation)
                    
                    # 清空当前会话
                    st.session_state.conversation = []
                    st.session_state.current_session_name = None
                    
                    # 刷新会话列表
                    st.session_state.sessions = load_session_history()
                    st.rerun()
                
                # 显示历史会话列表
                for idx, session in enumerate(st.session_state.sessions):
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col1:
                        if st.button(session["display_name"], key=f"session_{idx}", use_container_width=True):
                            # 保存当前会话
                            if len(st.session_state.conversation) > 0:
                                save_conversation(st.session_state.conversation)
                            
                            # 加载选中的会话
                            st.session_state.conversation = load_conversation(session["filename"])
                            st.session_state.current_session_name = session["filename"]
                            st.rerun()
                    
                    with col2:
                        # 删除按钮
                        if st.button("🗑️", key=f"delete_{idx}", help="删除此会话"):
                            if delete_conversation(session["filename"]):
                                st.success("会话已删除")
                                # 如果删除的是当前会话，清空当前会话
                                if st.session_state.current_session_name == session["filename"]:
                                    st.session_state.conversation = []
                                    st.session_state.current_session_name = None
                                
                                # 刷新会话列表
                                st.session_state.sessions = load_session_history()
                                st.rerun()
                
                # 如果没有历史会话，显示提示
                if not st.session_state.sessions:
                    st.info("没有历史会话记录，开始新的对话吧！")
                
                # 修复：确保use_knowledge_base变量被正确设置
                use_knowledge_base = st.session_state.use_knowledge_base
            else:
                # 显示设置面板
                use_knowledge_base = st.toggle("启用知识库", value=True)
                with st.expander("导入新模型", expanded=False):
                    new_name = st.text_input('输入模型名称')
                    if new_name:
                        with st.spinner('拉取模型中...'):
                            try:
                                ollama.pull(new_name)
                                if new_name not in st.session_state.model_names:
                                    st.session_state.model_names.append(new_name)
                                    st.success('模型拉取成功')
                            except Exception as e:
                                st.error(f'模型拉取失败，失败原因：{e}')
                                time.sleep(5)
                
                st.session_state.llm_model = st.selectbox(
                   "选择LLM模型：",
                    options=st.session_state.model_names,
                    index=0 if st.session_state.model_names else -1,
                    key="llm_model_select"
                )
                
                st.session_state.temperature = st.slider(
                    f"模型temperature设置",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.8,
                    key=f'temperature_slider'
                )
                
                if st.button("💾 立即保存当前会话"):
                    save_before_exit()
                    st.success("会话已保存！")
                    
                if use_knowledge_base:
                    st.session_state.kb_params = create_knowledge_base_settings()
        
        # 添加齿轮按钮到右下角
        st.markdown('<div class="settings-button">', unsafe_allow_html=True)
        if st.button("⚙️", help="切换设置面板", key="settings_button"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 创建主查询区 - 修复查询功能
    create_query_section(use_knowledge_base)

def create_control_buttons():
    """创建控制按钮"""
    col_reset, col_apply = st.columns(2)
    with col_reset:
        if st.button("重置"):
            reset_session_state()
            st.rerun()
    
    with col_apply:
        st.button("应用", type="primary", key="run_query_button")

def create_knowledge_base_settings() -> Dict[str, Any]:
    """创建知识库相关设置"""
    query_mode = st.selectbox(
        "查询模式选择",
        ["hybrid", "naive", "local", "global"],
        index=0,
        key="query_mode_select"
    )

    params = {}
    with st.expander("召回数量设置", expanded=False):
        params['entity_count'] = create_recall_settings("实体", "entity", 5000)
        params['relation_count'] = create_recall_settings("关系", "relation", 5000)
        params['doc_count'] = create_recall_settings("文档块", "doc", 1000)

    with st.expander("文件上传", expanded=False):
        create_file_upload_section(WORKING_DIR1, UPLOAD_FOLDER)

    params['custom_work_folder'] = st.text_input(
        "请输入参考知识库路径（或留空使用默认路径）:",
        value=WORKING_DIR
    )
    params['query_mode'] = query_mode

    create_control_buttons()
    return params



def create_query_section(use_knowledge_base: bool):
    """创建查询区域"""
    # 修改气泡文字颜色
    st.markdown("""
    <style>
    /* 新增底部容器样式 */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        z-index: 1000;
        box-shadow: 0 -2px 15px rgba(0,0,0,0.1);
    }
    
    /* 调整输入框容器宽度 */
    .query-input {
        width: calc(100% - 16rem) !important;  /* 减去侧边栏宽度 */
        margin: 0 auto;
    }
    
    /* 修复滚动条问题 */
    .stApp {
        padding-bottom: 6rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 显示历史对话（从HISTORY_SHOW.py迁移过来）
    if "conversation" in st.session_state and len(st.session_state.conversation) > 0:
        for idx, exchange in enumerate(st.session_state.conversation):
            # 用户提问气泡
            with st.chat_message("user"):
                st.markdown(f'<div class="user-bubble">{exchange["question"]}</div>', unsafe_allow_html=True)
            
            # 模型回答气泡
            with st.chat_message("assistant"):
                st.markdown(f'<div class="bot-bubble">{exchange["answer"]}</div>', unsafe_allow_html=True)
                
                # 知识参考
                with st.expander(f"📚 查看参考知识"):
                    if exchange["knowledge"] == "模型没有根据知识库做出的直接回答":
                        st.info(exchange["knowledge"])
                    else:
                        st.markdown("### 📊 实体")
                        display_entities(exchange["knowledge"], float('inf'))  # 使用无限值显示所有实体
                        st.markdown("### 🔄 关系")
                        display_relationships(exchange["knowledge"], float('inf'))
                        st.markdown("### 📄 文档")
                        display_sources(exchange["knowledge"], float('inf'))
    
    # 将输入框固定在底部（参考HISTORY_SHOW的布局）
    input_container = st.container()
    with input_container:
        st.markdown('<div class="query-input">', unsafe_allow_html=True)
        col_query, col_button = st.columns([8, 1])
        with col_query:
            # 修复：使用session_state中的query_text作为默认值，而不是直接修改组件的值
            if 'query_submitted' in st.session_state and st.session_state.query_submitted:
                # 如果查询已提交，则清空输入框
                query = st.text_input(
                    "Enter your query",
                    key="query_input",
                    label_visibility="collapsed",
                    placeholder="请输入您的问题...",
                    value=""  # 使用空字符串作为值
                )
                # 重置提交标志
                st.session_state.query_submitted = False
            else:
                query = st.text_input(
                    "Enter your query",
                    key="query_input",
                    label_visibility="collapsed",
                    placeholder="请输入您的问题..."
                )
        with col_button:
            run_button = st.button("🚀", help="运行查询", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 修复查询逻辑
    if query and (run_button or 
                 (use_knowledge_base and st.session_state.get('run_query_button', False))):
        # 设置查询已提交标志，用于下次渲染时清空输入框
        st.session_state.query_submitted = True
        
        params = st.session_state.get('kb_params', {}) if use_knowledge_base else {}
        
        # 确保params包含必要的键
        if use_knowledge_base and 'custom_work_folder' not in params:
            params['custom_work_folder'] = WORKING_DIR
        if use_knowledge_base and 'query_mode' not in params:
            params['query_mode'] = "hybrid"
        if use_knowledge_base and ('entity_count' not in params or 'relation_count' not in params or 'doc_count' not in params):
            # 设置默认值
            params['entity_count'] = float('inf') if st.session_state.get('unlimited_entity', True) else st.session_state.get('entity_value', DEFAULT_RECALL_COUNT)
            params['relation_count'] = float('inf') if st.session_state.get('unlimited_relation', True) else st.session_state.get('relation_value', DEFAULT_RECALL_COUNT)
            params['doc_count'] = float('inf') if st.session_state.get('unlimited_doc', True) else st.session_state.get('doc_value', DEFAULT_RECALL_COUNT)
        
        # 处理查询
        process_query(query, use_knowledge_base, params)
        
        # 重新渲染页面以显示新消息
        st.rerun()

if __name__ == "__main__":
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    main()
