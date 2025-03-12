import os
import streamlit as st
import ollama
import time
from pathlib import Path

# 导入自定义模块
from core.session import register_autosave, load_session_history, load_current_session
from core.query import process_query
from ui.layout import init_session_state, create_sidebar, create_query_section
from utils.model_utils import get_model_names

# 常量定义
SESSION_DIR = Path("./temp/chat")  # 会话保存路径
SESSION_DIR.mkdir(exist_ok=True, parents=True)

# 在主函数开始处添加
from core.session import initialize_session_state

def main():
    """主程序入口"""
    st.set_page_config(page_title="LightRAG", page_icon="🔍")
    st.title("LightRAG Query Interface")
    
    # 添加自定义CSS样式，确保对话框文字为黑色
    st.markdown("""
    <style>
    /* 用户气泡样式 */
    .user-bubble {
        background-color: #e6f7ff;
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        color: black !important;
    }
    
    /* 机器人气泡样式 */
    .bot-bubble {
        background-color: #f0f0f0;
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        color: black !important;
    }
    
    /* 确保所有对话文本为黑色 */
    .stChatMessage div[data-testid="stMarkdownContainer"] p {
        color: black !important;
    }
    
    /* 修复滚动条问题 */
    .stApp {
        padding-bottom: 6rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化会话状态
    initialize_session_state()
    
    # 注册自动保存
    register_autosave()
    
    # 加载当前会话
    if "conversation" not in st.session_state:
        st.session_state.conversation = load_current_session()
    
    # 加载模型列表
    if 'model_names' not in st.session_state:
        try:
            model_names = get_model_names()
            if model_names:
                s = ["deepseek-chat"]
                s.extend(model_names)
                st.session_state.model_names = s
            else:
                # 如果无法获取模型列表，使用默认值
                st.session_state.model_names = ["deepseek-chat", "llama2", "qwen"]
                st.warning("无法获取模型列表，使用默认模型列表。")
        except Exception as e:
            st.session_state.model_names = ["deepseek-chat", "llama2", "qwen"]
            st.warning(f"获取模型列表时出错: {str(e)}。使用默认模型列表。")
    
    # 创建侧边栏并获取是否使用知识库
    use_knowledge_base,use_autorag_base = create_sidebar()
    
    # 创建主查询区
    create_query_section(use_knowledge_base,use_autorag_base)

if __name__ == "__main__":
    main()