import io
import csv
import pandas as pd
import streamlit as st
import re

from utils.csv_utils import clean_markdown_csv, fix_csv_text, parse_csv_with_pipe_quotechar

def display_entities(context: str, entity_count: float):
    """显示实体信息"""
    if "-----实体-----" in context:
        try:
            entities_text = context.split("-----实体-----")[1].split("-----关系-----")[0]
            # 添加预处理步骤
            entities_text = entities_text.strip()
            if not entities_text:
                st.warning("没有找到实体数据")
                return
            
            # 检查是否是CSV格式
            if not any(c in entities_text for c in [',', '\t', '|', '+']):
                st.text(entities_text)
                return
            
            csv_text_cleaned = clean_markdown_csv(entities_text)
            final_entities_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
            
            # 尝试不同的分隔符
            separators = ['+', ',', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.StringIO(final_entities_text),
                        sep=sep,
                        quotechar=None,
                        quoting=csv.QUOTE_NONE,
                        engine="python",
                        on_bad_lines='skip'  # 跳过有问题的行
                    )
                    if not df.empty:
                        break
                except Exception:
                    continue
            
            if df is not None and not df.empty:
                st.dataframe(df)
            else:
                st.text(entities_text)
                
        except Exception as e:
            st.error(f"解析实体数据时出错: {str(e)}")

def display_relationships(context: str, relation_count: float):
    """显示关系信息"""
    if "-----关系-----" in context:
        try:
            relationships_text = context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
            relationships_df = pd.read_csv(io.StringIO(relationships_text))
            st.dataframe(relationships_df)
        except Exception as e:
            st.text(f"Error parsing relationships: {e}")
            st.text(relationships_text)

def display_sources(context: str, doc_count: float):
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