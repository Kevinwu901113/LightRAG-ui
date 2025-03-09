import os
import json
import uuid
import time
import streamlit as st
import atexit
from datetime import datetime
from typing import List, Dict, Any

# 会话文件存储目录
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_sessions")
# 当前会话文件
CURRENT_SESSION = os.path.join(SESSIONS_DIR, "current_session.json")

# 确保会话目录存在
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

def initialize_session_state():
    """初始化所有必要的会话状态变量"""
    # 初始化会话相关变量
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    # 添加 current_session_name 初始化
    if 'current_session_name' not in st.session_state:
        st.session_state.current_session_name = None
    
    # 初始化其他必要变量
    if 'sessions' not in st.session_state:
        st.session_state.sessions = load_session_history()
    
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    
    if 'use_knowledge_base' not in st.session_state:
        st.session_state.use_knowledge_base = True
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.8
    
    # 如果没有会话，自动创建一个新会话
    if not st.session_state.conversation and not st.session_state.current_session_id:
        # 尝试加载当前会话
        current_session = load_current_session()
        if current_session:
            st.session_state.conversation = current_session
            # 从会话列表中找到对应的会话ID
            for session in st.session_state.sessions:
                if session.get("is_current", False):
                    st.session_state.current_session_id = session.get("filename")
                    st.session_state.current_session_name = session.get("filename")
                    break
        else:
            # 如果没有当前会话，创建一个新会话
            from core.query import create_new_session
            create_new_session()
    # 初始化模型相关变量
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = 'deepseek-chat'  # 默认模型
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7  # 默认温度
    
    # 初始化知识库相关变量
    if 'use_knowledge_base' not in st.session_state:
        st.session_state.use_knowledge_base = True  # 默认使用知识库
    
    # 初始化UI相关变量
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    
    if 'query_submitted' not in st.session_state:
        st.session_state.query_submitted = False
    
    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = time.time()

def generate_session_id() -> str:
    """生成唯一的会话ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_conversation(conversation: List[Dict[str, Any]], session_id: str = None) -> bool:
    """保存对话到文件"""
    if not session_id:
        session_id = generate_session_id()
    
    # 确保文件名安全
    session_id = session_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    # 构建文件路径
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存会话失败: {str(e)}")
        return False

def load_conversation(session_id: str) -> List[Dict[str, Any]]:
    """加载指定的会话"""
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载会话失败: {str(e)}")
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

def delete_conversation(session_id: str) -> bool:
    """删除指定的会话"""
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    
    if not os.path.exists(file_path):
        return False
    
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"删除会话失败: {str(e)}")
        return False

# 新增函数
def load_current_session() -> List[Dict[str, Any]]:
    """加载当前会话"""
    if os.path.exists(CURRENT_SESSION):
        try:
            with open(CURRENT_SESSION, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载当前会话失败: {str(e)}")
    return []

def register_autosave():
    """注册自动保存功能"""
    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = time.time()
    
    # 检查是否需要自动保存（例如每60秒保存一次）
    current_time = time.time()
    if current_time - st.session_state.last_save_time > 60:  # 60秒自动保存一次
        if 'conversation' in st.session_state and len(st.session_state.conversation) > 0:
            if 'current_session_id' in st.session_state and st.session_state.current_session_id:
                save_conversation(st.session_state.conversation, st.session_state.current_session_id)
            else:
                # 如果没有当前会话ID，创建一个新的
                new_session_id = generate_session_id()
                st.session_state.current_session_id = new_session_id
                save_conversation(st.session_state.conversation, new_session_id)
            
            # 更新最后保存时间
            st.session_state.last_save_time = current_time
            
            # 同时保存到current_session.json
            try:
                with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存当前会话失败: {str(e)}")
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
            
            # 同时保存到current_session.json
            with open(CURRENT_SESSION, "w", encoding="utf-8") as f:
                json.dump(st.session_state.conversation, f, ensure_ascii=False, indent=2)
            
            print("会话已在退出前保存")
    except Exception as e:
        print(f"退出前保存会话失败: {str(e)}")

# 注册退出时的保存函数
atexit.register(save_before_exit)