import streamlit as st
from typing import Dict, Any, Optional
import time
import json

from core.session import load_session_history, load_conversation, save_conversation, delete_conversation, save_before_exit
from core.display import display_entities, display_relationships, display_sources
from core.query import process_query
from ui.components import create_recall_settings, create_file_upload_section, create_knowledge_base_settings
from utils.model_utils import get_model_names

# 常量定义
WORKING_DIR = "./dickens1"
WORKING_DIR1 = "tem"
UPLOAD_FOLDER = "uploads"
DEFAULT_RECALL_COUNT = 5

# 恢复init_session_state函数，因为app.py依赖它
def init_session_state():
    """初始化会话状态"""
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    if "use_knowledge_base" not in st.session_state:
        st.session_state.use_knowledge_base = True
    
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_session_history()
    
    if "current_session_name" not in st.session_state:
        st.session_state.current_session_name = None
    
    if "query_submitted" not in st.session_state:
        st.session_state.query_submitted = False
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.8
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

def display_conversation():
    """显示当前会话的对话历史"""
    # 添加自定义CSS样式
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
    
    # 检查是否有对话历史
    if 'conversation' not in st.session_state or not st.session_state.conversation:
        st.toast("没有对话历史，请开始一个新的对话。", icon="ℹ️")
        return
    
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
                        # 移除重复显示的原始知识库内容
                        # st.markdown(message.get("knowledge", ""))
# 移除重复的init_session_state函数，使用core.session中的initialize_session_state

def reset_session_state():
    """重置会话状态"""
    keys_to_delete = [
        'entity_value', 'relation_value', 'doc_value',
        'use_knowledge_base', 'query_mode', 'llm_model',
        'unlimited_entity', 'unlimited_relation', 'unlimited_doc',
        'show_settings', 
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

def refresh_sidebar():
    """刷新侧边栏会话列表"""
    # 更新会话列表
    st.session_state.sessions = load_session_history()
    # 强制重新运行应用
    st.rerun()

# 添加create_sidebar函数
def create_sidebar():
    """创建侧边栏"""
    with st.sidebar:
        # 创建一个容器用于显示主要内容
        main_content = st.container()
        
        with main_content:
            if not st.session_state.show_settings:
                # 显示历史会话区域
                st.markdown("### 历史会话")
                
                # 确保会话列表是最新的
                st.session_state.sessions = load_session_history()
                
                # 新建会话按钮
                if st.button("➕ 新建会话", key="new_session"):
                    # 清空当前会话，创建全新空白状态
                    st.session_state.conversation = []
                    st.session_state.current_session_id = None
                    st.session_state.current_session_name = None
                    st.rerun()
                
                # 显示历史会话列表
                for idx, session in enumerate(st.session_state.sessions):
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col1:
                        if st.button(session["display_name"], key=f"session_{idx}", use_container_width=True):
                            # 加载选中的会话，完全替换当前会话
                            st.session_state.conversation = load_conversation(session["filename"])
                            st.session_state.current_session_id = session["filename"]
                            st.session_state.current_session_name = session["filename"]
                            st.rerun()
                    
                    with col2:
                        # 删除按钮
                        # 在删除按钮部分
                        if st.button("🗑️", key=f"delete_{idx}", help="删除此会话"):
                            # 修复：检查 current_session_name 是否存在
                            current_name = st.session_state.get("current_session_name")
                            if delete_conversation(session["filename"]):
                                st.success("会话已删除")
                                # 注意：删除当前会话的逻辑已经移到 delete_conversation 函数中
                                st.rerun()
                    
                    with col3:
                        # 导出按钮
                        if st.button("💾", key=f"export_{idx}", help="导出此会话"):
                            conversation = load_conversation(session["filename"])
                            # 创建导出文件
                            export_data = json.dumps(conversation, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="下载",
                                data=export_data,
                                file_name=f"export_{session['filename']}",
                                mime="application/json",
                                key=f"download_{idx}"
                            )
                
                # 如果没有历史会话，显示提示
                if not st.session_state.sessions:
                    st.toast("没有历史会话记录，开始新的对话吧！", icon="ℹ️")
                
                # 确保use_knowledge_base变量被正确设置
                use_knowledge_base = st.session_state.use_knowledge_base
            else:
                # 显示设置面板
                use_knowledge_base = st.toggle("启用知识库", value=True)
                with st.expander("导入新模型", expanded=False):
                    new_name = st.text_input('输入模型名称')
                    if new_name:
                        with st.spinner('拉取模型中...'):
                            try:
                                import ollama
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
                    st.toast("会话已保存！", icon="✅")
                st.session_state.kb_params={'use_autorag_base':False}    
                if use_knowledge_base:
                    st.session_state.kb_params = create_knowledge_base_settings()
        
        # 添加齿轮按钮到右下角
        st.markdown('<div class="settings-button">', unsafe_allow_html=True)
        if st.button("⚙️", help="切换设置面板", key="settings_button"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        if st.session_state.kb_params!={}:
            return use_knowledge_base,st.session_state.kb_params["use_autorag_base"]
        return use_knowledge_base,False

# 其他函数保持不变
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
    params['use_autorag_base'] = st.toggle("启用autoRAG功能", value=True)
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
def create_query_section(use_knowledge_base=True,use_autoRAG_base=True):
    """创建查询区域"""
    # 添加底部固定输入框样式
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
    
    # 确保有活跃的会话
    from core.query import ensure_active_session
    ensure_active_session()
    
    # 显示对话历史
    display_conversation()
    
    # 将输入框固定在底部
    input_container = st.container()
    with input_container:
        st.markdown('<div class="query-input">', unsafe_allow_html=True)
        col_query, col_button = st.columns([8, 1])
        
        # 定义一个回调函数，用于处理查询提交
        def handle_submit():
            """处理用户提交的查询"""
            if st.session_state.query_input:
                current_query = st.session_state.query_input
                use_knowledge_base = st.session_state.use_knowledge_base
                
                # 创建params参数字典
                params = {
                    "temperature": st.session_state.temperature,
                    "custom_work_folder": "dickens1"  # 或其他适当的工作文件夹
                }
                
                # 处理查询
                process_query(current_query, use_knowledge_base, params)
                
                # 清空输入框
                st.session_state.query_input = ""
                
                # 强制刷新侧边栏
                refresh_sidebar()
                
                # 标记查询已提交
                st.session_state.query_submitted = True
                
                # 设置刷新侧边栏标志
                st.session_state.refresh_sidebar_needed = True
                
                # 使用 Streamlit 的方式清空输入框 - 通过设置一个标记
                st.session_state.clear_input = True
        
        with col_query:
            # 使用 key 确保输入框状态正确更新
            query = st.text_input(
                "Enter your query",
                key="query_input",
                label_visibility="collapsed",
                placeholder="请输入您的问题...",
                on_change=handle_submit
            )
        
        with col_button:
            # 发送按钮 - 点击时触发相同的回调
            if st.button("🚀", help="发送", use_container_width=True, key="send_button"):
                handle_submit()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 如果有待处理的查询，处理它
    if 'current_query' in st.session_state and st.session_state.get('query_submitted', False):
        # 重置标志，防止重复处理
        st.session_state.query_submitted = False
        # 清除当前查询
        if 'current_query' in st.session_state:
            del st.session_state.current_query
        
        # 触发页面重新加载以显示新消息
        st.rerun()