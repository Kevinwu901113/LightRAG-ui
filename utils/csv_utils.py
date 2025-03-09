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