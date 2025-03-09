import os
import shutil
import textract
from typing import List
import streamlit as st

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

def reset_corrupted_kb_files(work_folder: str):
    """重置损坏的知识库文件"""
    import json
    json_files = [
        os.path.join(work_folder, f)
        for f in os.listdir(work_folder)
        if f.endswith('.json')
    ]
    for json_file in json_files:
        with open(json_file, 'w') as f:
            json.dump({}, f)