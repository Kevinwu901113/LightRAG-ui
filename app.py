import os
import streamlit as st
import ollama
import time
from pathlib import Path

# 导入自定义模块
from core.session import load_session_history
from core.query import process_query
from ui.layout import init_session_state, create_sidebar, create_query_section
from utils.model_utils import get_model_names
# 导入显示函数
from core.display import display_entities, display_relationships, display_sources

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
    
    try:
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
            except Exception as e:
                st.warning(f"获取模型列表时出错: {str(e)}")
        
        # 创建侧边栏并获取是否使用知识库
        use_knowledge_base = create_sidebar()
        
        # 显示历史对话
        display_conversation()
        
        # 创建主查询区
        create_query_section(use_knowledge_base)
    except Exception as e:
        st.error(f"应用程序运行时出错: {str(e)}")

def display_conversation():
    """显示当前会话的对话历史"""
    
    # 确保会话状态中有 conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    # 遍历对话历史并显示
    for message in st.session_state.conversation:
        # 显示用户问题
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{message.get("question", "")}</div>', unsafe_allow_html=True)
        
        # 显示AI回答
        if message.get("answer"):
            with st.chat_message("assistant"):
                st.markdown(f'<div class="bot-bubble">{message.get("answer", "")}</div>', unsafe_allow_html=True)
                
                # 如果有知识库内容，显示展开/折叠选项
                if message.get("knowledge"):
                    with st.expander("📚 查看知识库引用"):
                        if message["knowledge"] == "模型没有根据知识库做出的直接回答":
                            st.info(message["knowledge"])
                        else:
                            st.markdown("### 📊 实体")
                            display_entities(message["knowledge"], float('inf'))  # 使用无限值显示所有实体
                            st.markdown("### 🔄 关系")
                            display_relationships(message["knowledge"], float('inf'))
                            st.markdown("### 📄 文档")
                            display_sources(message["knowledge"], float('inf'))

if __name__ == "__main__":
    main()