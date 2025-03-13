import json
import asyncio
import os
import random  # 添加缺失的导入
import string  # 添加缺失的导入
import streamlit as st
from typing import Dict, Any
from datetime import datetime
from lightrag import QueryParam
from query import insert1, query1, prompt_1, prompt_2, prompt_3,autorag,direct_query

from utils.csv_utils import clean_markdown_csv, fix_csv_text
from utils.file_utils import reset_corrupted_kb_files
from core.display import display_entities, display_relationships, display_sources
from core.session import save_conversation, load_conversation, load_session_history, generate_session_id

# 移除对CURRENT_SESSION的引用

def ensure_active_session():
    """确保有一个活跃的会话，如果没有则创建一个新会话"""
    # 检查是否有会话列表
    if 'sessions' not in st.session_state:
        st.session_state.sessions = load_session_history()
    
    # 检查是否有当前会话
    if 'current_session_id' not in st.session_state or not st.session_state.current_session_id:
        # 如果有历史会话，加载最新的一个
        if st.session_state.sessions:
            latest_session = st.session_state.sessions[0]
            st.session_state.current_session_id = latest_session["filename"]
            st.session_state.conversation = load_conversation(latest_session["filename"])
        else:
            # 没有任何会话，创建一个新会话
            new_session_id = generate_session_id()
            st.session_state.current_session_id = new_session_id
            st.session_state.conversation = []
            # 更新会话列表
            st.session_state.sessions = load_session_history()

def create_new_session():
    """创建一个全新的空白会话"""
    # 清空当前会话
    st.session_state.conversation = []
    
    # 生成新的会话ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.hexdigits.lower(), k=8))
    st.session_state.current_session_id = f"session_{timestamp}_{random_suffix}"
    st.session_state.current_session_name = None
    
    # 不立即保存，等到有内容时再保存
    return st.session_state.current_session_id
def process_kb_query(query: str, model_index: int, params: Dict[str, Any], use_autoRAG_base=True):
    """处理基于知识库的查询"""
    try:
        # 获取工作目录和模型名称，添加默认值处理
        if not params:
            params = {'custom_work_folder': 'dickens1', 'use_autorag_base': False}
        work_folder = params.get('custom_work_folder', params.get('work_folder', './knowledge_base'))
        
        # 检查工作目录是否存在
        if not os.path.exists(work_folder):
            # 尝试使用dickens1目录作为备选
            if os.path.exists('./knowledge_base'):
                work_folder = './knowledge_base'
                print(f"原工作目录不存在，切换到备选目录: {work_folder}")
            else:
                return f"知识库查询失败: 知识库目录 '{work_folder}' 不存在，请先上传文件创建知识库。", ""
        
        # 检查知识库文件是否存在
        graphml_path = os.path.join(work_folder, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml_path):
            return f"知识库查询失败: 知识库文件 '{graphml_path}' 不存在，请先上传文件创建知识库。", ""
        
        model_name = st.session_state.llm_model
        temperature = st.session_state.temperature
        
        # 创建查询参数
        query_param = QueryParam(mode="hybrid", only_need_context=False)
        
        print(f"使用知识库目录: {work_folder}")
        
        # 调用查询函数
        result = query1(
            working_dir=work_folder,
            query=query,
            model_name=model_name,
            i=model_index,
            param=query_param,
            use_kb=True,
            temperature=temperature
        )
        query_param = QueryParam(mode="hybrid", only_need_context=True)
        knowledge = query1(
            working_dir=work_folder,
            query=query,
            model_name=model_name,
            i=model_index,
            param=query_param,
            use_kb=True,
            temperature=temperature
        )
        return result, knowledge
        # # 解析结果
        # if isinstance(result, dict):
        #     answer = result.get('answer', '')
        #     knowledge = result.get('knowledge', '')
            
        #     # 处理知识库内容
        #     if knowledge:
        #         # 清理CSV格式的知识
        #         knowledge = clean_markdown_csv(knowledge)
        #         knowledge = fix_csv_text(knowledge)
        #     if use_autoRAG_base:
        #         success,ans_his=autorag(query,answer,knowledge,5,model_name,temperature)
        #         answer=ans_his["answer"]
        #         history=ans_his["history"]
        #     return answer, knowledge
        # else:
        #     # 如果结果不是字典，直接返回
        #     return result, ""
            
    except json.JSONDecodeError as e:
        handle_kb_error(e, work_folder)
        return f"知识库查询失败: {str(e)}", ""
    except Exception as e:
        import traceback
        error_msg = f"知识库查询失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"知识库查询失败: {str(e)}", ""

def process_direct_query(query: str, model_index: int, temperature: float = 0.8):
    """处理直接查询"""
    try:
        # 获取模型名称
        model_name = st.session_state.llm_model
        
        # 修改异步调用方式，确保在新的事件循环中运行
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 在事件循环中运行异步函数
        result = loop.run_until_complete(direct_query(query, model_name, temperature))
        return result
    except Exception as e:
        import traceback
        error_msg = f"直接查询失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"直接查询失败: {str(e)}"

def process_query(query: str, use_kb: bool, params: Dict[str, Any]):
    """处理查询请求"""
    model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1

    # 如果当前没有会话ID，创建一个新的会话ID
    if not st.session_state.current_session_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.hexdigits.lower(), k=8))
        st.session_state.current_session_id = f"session_{timestamp}_{random_suffix}"
    
    if use_kb:
        try:
            answer, context, = process_kb_query(query, model_index, params)
            
            # 显示结果
            st.header("生成的回答")
            st.write(answer)
            
            # 保存到会话历史
            st.session_state.conversation.append({
                "question": query,
                "answer": answer,
                "knowledge": context,
                "timestamp": datetime.now().isoformat()
            })
            
        except json.JSONDecodeError as e:
            st.error(f"知识库文件损坏！请尝试重新上传文件创建知识库。错误详情：{str(e)}")
        except Exception as e:
            st.error(f"查询过程中发生错误：{str(e)}")
    else:
        # 直接查询模式
        answer = process_direct_query(query, model_index, st.session_state.temperature)
        
        # 保存到会话历史
        st.session_state.conversation.append({
            "question": query,
            "answer": answer,
            "knowledge": "模型没有根据知识库做出的直接回答",
            "timestamp": datetime.now().isoformat()
        })
    
    # 每次查询后保存会话
    if st.session_state.current_session_id:
        save_conversation(st.session_state.conversation, st.session_state.current_session_id)
    else:
        new_session_id = generate_session_id()
        st.session_state.current_session_id = new_session_id
        save_conversation(st.session_state.conversation, new_session_id)
    
    # 更新会话列表并设置刷新标志
    st.session_state.sessions = load_session_history()
    st.session_state.refresh_sidebar_needed = True
    
    return True
def handle_kb_error(error: json.JSONDecodeError, work_folder: str):
    """处理知识库错误"""
    st.toast("知识库文件损坏！请尝试重新上传文件创建知识库。", icon="⚠️")
    st.toast(f"错误详情：{str(error)}", icon="⚠️")
    try:
        reset_corrupted_kb_files(work_folder)
        st.toast("已重置损坏的知识库文件。请重新上传文件创建知识库。", icon="ℹ️")
    except Exception as e:
        st.toast(f"重置知识库文件失败: {str(e)}", icon="⚠️")