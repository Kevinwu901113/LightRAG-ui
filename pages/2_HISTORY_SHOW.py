import streamlit as st
import json
from datetime import datetime
import os
import re
from pathlib import Path
import csv
import io
import pandas as pd
# 创建存储对话记录的目录
SESSION_DIR = Path("chat_sessions")
SESSION_DIR.mkdir(exist_ok=True)

# ==================== 数据存储逻辑 ====================

def load_session(filename):
    """加载历史会话记录"""
    try:
        with open(SESSION_DIR / filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载会话失败: {str(e)}")
        return None

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

def display_entities(context: str,id:int):
    """显示实体信息"""
    if "-----实体-----" in context:
        
            try:
                entities_text = context.split("-----实体-----")[1].split("-----关系-----")[0]
                csv_text_cleaned = clean_markdown_csv(entities_text)
                final_entities_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)

                entities_df = pd.read_csv(
                    io.StringIO(final_entities_text),
                    sep='+',
                    quotechar=None,
                    quoting=csv.QUOTE_NONE,
                    engine="python"
                )
                st.dataframe(entities_df)
            except Exception as e:
                st.text(f"Error parsing entities: {e}")
                st.text(entities_text)

def display_relationships(context: str,id:int):
    """显示关系信息"""
    if "-----关系-----" in context:
        
            try:
                relationships_text = context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
                relationships_df = pd.read_csv(io.StringIO(relationships_text))
                st.dataframe(relationships_df)
            except Exception as e:
                st.text(f"Error parsing relationships: {e}")
                st.text(relationships_text)

def display_sources(context: str,id:int):
    """显示来源信息"""
    if "-----信息来源-----" in context:
       
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

# ==================== 页面布局 ====================
def main():
    st.set_page_config(
        page_title="对话历史查看器",
        page_icon="💬",
        layout="wide"
    )
    
    # 自定义CSS样式
    st.markdown("""
    <style>
    .user-bubble {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .bot-bubble {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .knowledge-ref {
        border-left: 3px solid #4CAF50;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==================== 侧边栏 ====================
    st.sidebar.title("会话管理")
    
    # 显示已有会话文件
    session_files = [f.name for f in SESSION_DIR.glob("*.json")]
    selected_file = st.sidebar.selectbox(
        "选择历史会话",
        session_files,
        index=len(session_files)-1 if session_files else 0
    )
    
    # ==================== 主界面 ====================
    st.title("💬 对话历史查看器")
    
    if selected_file:
        session_data = load_session(selected_file)
        
        if session_data:
            for idx, exchange in enumerate(session_data):
                # 用户提问 (右侧)
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col2:
                        st.markdown(
                            f'<div class="user-bubble">'
                            f'<strong>👤 用户提问：</strong><br>{exchange["question"]}'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                
                # 模型回答 (左侧)
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f'<div class="bot-bubble">'
                            f'<strong>🤖 模型回答：</strong><br>{exchange["answer"]}'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # 可展开的知识参考
                        with st.expander(f"📚 查看参考知识- 第{idx+1}条"):
                            if exchange["knowledge"]=="模型没有根据知识库做出的直接回答":
                               
                                st.markdown(
                                    f'<div class="knowledge-ref">{exchange["knowledge"]}</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                display_entities(exchange["knowledge"],idx+1)
                                display_relationships(exchange["knowledge"],idx+1)
                                display_sources(exchange["knowledge"],idx+1)
                
                st.markdown("---")  # 对话分隔线

    else:
        st.info("⏳ 暂无历史会话记录，请先进行对话")

# ==================== 使用示例 ==================== 
if __name__ == "__main__":
    # 模拟保存一个示例会话
    
    main()