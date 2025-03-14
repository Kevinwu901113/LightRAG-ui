import os
import json
import uuid
import time
import random
import string
import streamlit as st
import atexit
from datetime import datetime
from typing import List, Dict, Any

# 会话文件存储目录
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_sessions")

# 确保会话目录存在
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

def initialize_session_state():
    """初始化会话状态"""
    # 定义默认值字典
    default_values = {
        'conversation': [],  # 初始为空列表
        'current_session_id': None,
        'current_session_name': None,
        'sessions': load_session_history(),
        'show_settings': False,
        'use_knowledge_base': True,
        'use_autorag_base': False,  # 添加AutoRAG开关默认值
        'temperature': 0.7,
        'llm_model': 'deepseek-chat',
        'query_submitted': False,
        'kb_params': {'custom_work_folder': 'dickens1', 'use_autorag_base': False},  # 确保kb_params有默认值
    }
    
    # 为每个默认值设置session state
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # 始终创建一个新的空白会话，不尝试加载已有会话
    if not st.session_state.get("current_session_id"):
        # 生成新的会话ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.hexdigits.lower(), k=8))
        st.session_state.current_session_id = f"session_{timestamp}_{random_suffix}"
        st.session_state.conversation = []  # 确保对话为空

def generate_session_id() -> str:
    """生成唯一的会话ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_conversation(conversation, filename=None):
    """保存会话到文件"""
    try:
        # 如果会话为空，不保存
        if not conversation:
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
        
        file_path = os.path.join(SESSIONS_DIR, f"{filename}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        # 保存后刷新会话列表
        if 'sessions' in st.session_state:
            st.session_state.sessions = load_session_history()
            
        return filename
    except Exception as e:
        st.toast(f"保存会话失败: {str(e)}", icon="⚠️")
        return None

def delete_conversation(filename):
    """删除指定的会话文件"""
    try:
        file_path = os.path.join(SESSIONS_DIR, f"{filename}.json")  # 修改为 SESSIONS_DIR
        if os.path.exists(file_path):
            os.remove(file_path)  # 使用 os.remove 替代 file_path.unlink()
            
            # 删除后刷新会话列表
            if 'sessions' in st.session_state:
                st.session_state.sessions = load_session_history()
            
            # 如果删除的是当前会话，创建一个新的未保存会话
            if 'current_session_name' in st.session_state and st.session_state.current_session_name == filename:
                st.session_state.conversation = []
                st.session_state.current_session_name = None
                
                # 生成新的会话ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_suffix = ''.join(random.choices(string.hexdigits.lower(), k=8))
                st.session_state.current_session_id = f"session_{timestamp}_{random_suffix}"
                
            return True
        return False
    except Exception as e:
        st.toast(f"删除会话失败: {str(e)}", icon="⚠")
        return False



def load_conversation(filename):
    """加载指定的会话文件"""
    try:
        file_path = os.path.join(SESSIONS_DIR, f"{filename}.json")  # 修改为 SESSIONS_DIR
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        st.toast(f"加载会话失败: {str(e)}", icon="⚠️")
        return []

def load_session_history() -> List[Dict[str, str]]:
    """加载所有会话历史"""
    sessions = []
    
    if not os.path.exists(SESSIONS_DIR):
        return sessions
    
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json") and filename != "current_session.json":
            file_path = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    conversation = json.load(f)
                
                # 获取会话的第一个问题作为显示名称
                display_name = "新会话"
                if conversation and len(conversation) > 0:
                    first_question = conversation[0].get("question", "")
                    if first_question:
                        # 截取前20个字符作为显示名称
                        display_name = first_question[:20] + ("..." if len(first_question) > 20 else "")
                
                # 获取文件修改时间
                mod_time = os.path.getmtime(file_path)
                
                sessions.append({
                    "filename": filename.replace(".json", ""),
                    "display_name": display_name,
                    "modified_time": mod_time
                })
            except Exception as e:
                print(f"加载会话 {filename} 失败: {str(e)}")
    
    # 按修改时间倒序排序
    sessions.sort(key=lambda x: x["modified_time"], reverse=True)
    return sessions

def save_before_exit():
    """在应用程序退出前保存当前会话"""
    try:
        if 'conversation' in st.session_state and len(st.session_state.conversation) > 0:
            if 'current_session_id' in st.session_state and st.session_state.current_session_id:
                save_conversation(st.session_state.conversation, st.session_state.current_session_id)
            else:
                # 如果没有当前会话ID，创建一个新的
                new_session_id = generate_session_id()
                save_conversation(st.session_state.conversation, new_session_id)
            
            print("会话已在退出前保存")
    except Exception as e:
        print(f"退出前保存会话失败: {str(e)}")
