import re
import csv
import io
import pandas as pd

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

def process_irregular_line_breaks(csv_text: str) -> str:
    """
    处理包含不规则换行的CSV文本
    
    Args:
        csv_text (str): 输入的CSV文本
    
    Returns:
        str: 处理后的CSV文本
    """
    # 分割成行
    lines = csv_text.strip().split('\n')
    
    # 跳过标题行
    if lines and 'id' in lines[0] and 'entity' in lines[0]:
        header = lines[0]
        lines = lines[1:]
    else:
        header = ""
    
    # 合并记录
    records = []
    current_record = []
    quote_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 计算引号数量来判断是否在一个字段内
        quote_count += line.count('"')
        current_record.append(line)
        
        # 如果引号数量是偶数，且有逗号和数字结尾，可能是一条完整记录
        if quote_count % 2 == 0 and re.search(r',\s*\d+\s*$', line):
            record_text = ' '.join(current_record)
            records.append(record_text)
            current_record = []
            quote_count = 0
    
    # 处理最后一条可能未完成的记录
    if current_record:
        records.append(' '.join(current_record))
    
    # 重新组合成CSV文本
    result = header + "\n" + "\n".join(records) if header else "\n".join(records)
    return result

def has_irregular_line_breaks(csv_text: str) -> bool:
    """
    检查CSV文本是否包含不规则的换行
    
    Args:
        csv_text (str): 输入的CSV文本
    
    Returns:
        bool: 如果包含不规则换行则返回True，否则返回False
    """
    lines = csv_text.strip().split('\n')
    if len(lines) <= 1:
        return False
    
    # 跳过标题行
    if 'id' in lines[0] and 'entity' in lines[0]:
        lines = lines[1:]
    
    for line in lines:
        line = line.strip()
        # 如果行不是以数字开头，可能是不规则换行的一部分
        if line and not re.match(r'^\d+\s*,', line):
            return True
    
    return False

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