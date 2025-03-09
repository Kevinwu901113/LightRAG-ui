import os
import streamlit as st
from typing import Dict, Any

from query import insert1
from utils.file_utils import save_uploaded_file, extract_text_from_docs
from core.session import save_before_exit, save_conversation, delete_conversation, load_conversation, load_session_history

# 常量定义
DEFAULT_RECALL_COUNT = 5
WORKING_DIR = "./dickens1"
WORKING_DIR1 = "tem"
UPLOAD_FOLDER = "uploads"

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

def create_control_buttons():
    """创建控制按钮"""
    col_reset, col_apply = st.columns(2)
    with col_reset:
        if st.button("重置"):
            reset_session_state()
            st.rerun()
    
    with col_apply:
        st.button("应用", type="primary", key="run_query_button")

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