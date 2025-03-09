import json
import asyncio
import os
import streamlit as st
from typing import Dict, Any
from datetime import datetime
from lightrag import QueryParam
from query import insert1, query1, prompt_1, prompt_2, prompt_3, direct_query

from utils.csv_utils import clean_markdown_csv, fix_csv_text
from utils.file_utils import reset_corrupted_kb_files
from core.display import display_entities, display_relationships, display_sources
from core.session import save_conversation, load_conversation, load_session_history, CURRENT_SESSION, generate_session_id

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
    """创建一个全新的会话"""
    # 保存当前会话
    if 'conversation' in st.session_state and len(st.session_state.conversation) > 0:
        save_conversation(st.session_state.conversation)
    
    # 创建新会话
    new_session_id = generate_session_id()
    st.session_state.current_session_id = new_session_id
    # 同时设置 current_session_name
    st.session_state.current_session_name = new_session_id
    st.session_state.conversation = []
    
    # 更新会话列表
    st.session_state.sessions = load_session_history()
    return new_session_id
def process_kb_query(query: str, model_index: int, params: Dict[str, Any]):
    """处理基于知识库的查询"""
    try:
        # 获取工作目录和模型名称
        work_folder = params.get('custom_work_folder', params.get('work_folder', './temp'))
        
        # 检查工作目录是否存在
        if not os.path.exists(work_folder):
            # 尝试使用dickens1目录作为备选
            if os.path.exists('./dickens1'):
                work_folder = './dickens1'
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
        
        # 解析结果
        if isinstance(result, dict):
            answer = result.get('answer', '')
            knowledge = result.get('knowledge', '')
            
            # 处理知识库内容
            if knowledge:
                # 清理CSV格式的知识
                knowledge = clean_markdown_csv(knowledge)
                knowledge = fix_csv_text(knowledge)
            
            return answer, knowledge
        else:
            # 如果结果不是字典，直接返回
            return result, ""
            
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
        
        # 调用直接查询函数
        result = asyncio.run(direct_query(query, model_name, temperature))
        return result, ""
    except Exception as e:
        import traceback
        error_msg = f"直接查询失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"直接查询失败: {str(e)}", ""

def handle_kb_error(error: json.JSONDecodeError, work_folder: str):
    """处理知识库错误"""
    st.error("知识库文件损坏！请尝试重新上传文件创建知识库。")
    st.error(f"错误详情：{str(error)}")
    try:
        reset_corrupted_kb_files(work_folder)
        st.info("已重置损坏的知识库文件。请重新上传文件创建知识库。")
    except Exception as e:
        st.error(f"重置知识库文件失败: {str(e)}")

def process_query(query, use_knowledge_base=True):
    """处理用户查询"""
    try:
        # 确保有活跃的会话
        ensure_active_session()
        
        # 添加用户问题到会话
        timestamp = datetime.now().isoformat()
        user_message = {
            "question": query,
            "answer": "",
            "knowledge": "",
            "timestamp": timestamp
        }
        
        # 将用户问题添加到会话
        st.session_state.conversation.append(user_message)
        
        # 显示处理中的消息
        with st.spinner("正在处理您的问题..."):
            # 确定模型索引和参数
            model_index = 0 if st.session_state.llm_model == 'deepseek-chat' else 1
            temperature = st.session_state.temperature
            
            # 准备查询参数 - 修改这里，使用用户设置的知识库路径
            params = {
                'temperature': temperature
            }
            
            # 如果存在用户设置的知识库路径，使用它
            if 'kb_params' in st.session_state and 'custom_work_folder' in st.session_state.kb_params:
                params['custom_work_folder'] = st.session_state.kb_params['custom_work_folder']
            else:
                # 默认使用 dickens1 目录
                params['custom_work_folder'] = './dickens1'
                
            # 打印当前使用的知识库路径，便于调试
            print(f"使用知识库路径: {params.get('custom_work_folder', './temp')}")
            
            # 根据是否使用知识库选择不同的处理方式
            if use_knowledge_base:
                print(f"使用知识库处理查询: {query}")
                answer, knowledge = process_kb_query(query, model_index, params)
            else:
                print(f"直接使用模型处理查询: {query}")
                answer, knowledge = process_direct_query(query, model_index, temperature)
            
            # 更新会话中的回答
            st.session_state.conversation[-1]["answer"] = answer
            st.session_state.conversation[-1]["knowledge"] = knowledge
            
            # 保存会话
            if st.session_state.current_session_id:
                save_conversation(st.session_state.conversation, st.session_state.current_session_id)
            else:
                new_session_id = generate_session_id()
                st.session_state.current_session_id = new_session_id
                save_conversation(st.session_state.conversation, new_session_id)
                st.session_state.sessions = load_session_history()
            
            # 同时保存到当前会话文件
            try:
                with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存当前会话失败: {str(e)}")
        
        return True
    except Exception as e:
        import traceback
        print(f"处理查询时出错: {str(e)}")
        print(traceback.format_exc())
        st.error(f"处理查询时出错: {str(e)}")
        return False