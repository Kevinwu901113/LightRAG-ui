import io
import csv
import pandas as pd
import streamlit as st
import re

from utils.csv_utils import (
    clean_markdown_csv, 
    fix_csv_text, 
    parse_csv_with_pipe_quotechar, 
    process_irregular_line_breaks,
    has_irregular_line_breaks
)

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
            
            # 检查并处理不规则换行
            if has_irregular_line_breaks(csv_text_cleaned):
                csv_text_cleaned = process_irregular_line_breaks(csv_text_cleaned)
            
            final_entities_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
            
            # 预处理 + 分隔符的特殊情况
            processed_text = final_entities_text
            # 处理引号问题 - 将不成对的引号替换
            processed_text = re.sub(r'(?<!")"(?!")', '\'', processed_text)
            # 确保 + 符号在字段内容中不会被误解为分隔符
            processed_text = re.sub(r'"\+(?=[^"]*")', '"\\+', processed_text)
            
            # 尝试不同的分隔符
            separators = ["+", ',', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    if sep == "+":
                        # 使用更复杂的处理方法来处理 + 分隔符
                        lines = processed_text.split('\n')
                        rows = []
                        
                        for line in lines:
                            if line.strip():
                                # 预处理：保护 {} 内的内容，防止被分割
                                # 先替换 {} 内的 + 号为临时标记
                                protected_line = line
                                # 找出所有 {} 括号内的内容
                                brace_contents = re.findall(r'\{[^{}]*\}', line)
                                
                                # 为每个 {} 内容创建一个临时替代标记
                                for i, content in enumerate(brace_contents):
                                    # 替换 {} 内的 + 为临时标记
                                    safe_content = content.replace('+', '§§PLUS§§')
                                    # 在原始行中替换 {} 内容
                                    protected_line = protected_line.replace(content, safe_content, 1)
                                
                                # 现在可以安全地按 + 分割
                                fields = [f.strip() for f in protected_line.split('+')]
                                
                                # 恢复每个字段中的 + 号
                                fields = [f.replace('§§PLUS§§', '+') for f in fields]
                                
                                # 处理引号
                                fields = [f.strip('"\'') for f in fields if f.strip()]
                                
                                if fields:
                                    rows.append(fields)
                        
                        if rows:
                            # 确定最大列数 - 应该是固定的5列 (id, entity, type, description, rank)
                            expected_cols = 5
                            
                            # 处理每一行，确保只有5列
                            processed_rows = []
                            for row in rows:
                                if len(row) > expected_cols:
                                    # 如果列数超过5，将多余的列合并到description列
                                    new_row = row[:3]  # 保留id, entity, type
                                    # 合并description列及之后的所有列，直到rank列
                                    description = '+'.join(row[3:-1])
                                    new_row.append(description)
                                    new_row.append(row[-1])  # 添加rank列
                                    processed_rows.append(new_row)
                                else:
                                    # 如果列数不足5，填充空值
                                    while len(row) < expected_cols:
                                        row.append("")
                                    processed_rows.append(row)
                            
                            # 处理列名，确保没有重复
                            if processed_rows:
                                headers = ["id", "entity", "type", "description", "rank"]
                                
                                # 创建DataFrame，使用固定的列名
                                df = pd.DataFrame(processed_rows[1:], columns=headers)
                    else:
                        df = pd.read_csv(
                            io.StringIO(processed_text),
                            sep=sep,
                            quotechar='"',
                            escapechar='\\',
                            engine="python",
                            on_bad_lines='skip'  # 跳过有问题的行
                        )
                        
                        # 处理非 + 分隔符情况下的重复列名
                        if not df.empty:
                            # 重命名重复列
                            df.columns = pd.Series(df.columns).map(lambda x: f"{x}_{i}" if list(df.columns).count(x) > 1 and i > list(df.columns).index(x) else x for i in range(len(df.columns)))
                    
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    st.text(f"使用分隔符 {sep} 时出错: {str(e)}")
                    continue
            
            if df is not None and not df.empty:
                st.dataframe(df)
            else:
                st.error(f"解析实体数据时出错")
                st.text(entities_text)
                
        except Exception as e:
            st.error(f"解析实体数据时出错: {str(e)}")

def display_relationships(context: str, relation_count: float):
    """显示关系信息"""
    if "-----关系-----" in context:
        try:
            relationships_text = context.split("-----关系-----")[1].split("-----信息来源-----")[0].strip()
            # 添加预处理步骤
            if not relationships_text:
                st.warning("没有找到关系数据")
                return
            
            # 检查是否是CSV格式
            if not any(c in relationships_text for c in [',', '\t', '|', '+']):
                st.text(relationships_text)
                return
            
            csv_text_cleaned = clean_markdown_csv(relationships_text)
            
            # 检查并处理不规则换行
            if has_irregular_line_breaks(csv_text_cleaned):
                csv_text_cleaned = process_irregular_line_breaks(csv_text_cleaned)
            
            final_relationships_text = parse_csv_with_pipe_quotechar(csv_text_cleaned)
            
            # 尝试不同的分隔符
            separators = ['+', ',', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.StringIO(final_relationships_text),
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
                st.text(relationships_text)
                
        except Exception as e:
            st.error(f"解析关系数据时出错: {str(e)}")
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
