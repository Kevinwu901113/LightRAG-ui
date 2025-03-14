import csv
import re

def process_entities_file(input_file, output_file):
    """
    处理包含不规则换行的实体文件，更健壮的版本
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 跳过文件开头的空行和代码块标记
    if '```csv' in content:
        content = content.split('```csv', 1)[1]
        if '```' in content:
            content = content.split('```', 1)[0]
    
    # 分割成行
    lines = content.strip().split('\n')
    
    # 跳过标题行
    if lines and 'id' in lines[0] and 'entity' in lines[0]:
        header = lines[0]
        lines = lines[1:]
    
    # 合并记录
    records = []
    current_record = []
    in_record = False
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
    
    # 解析记录
    parsed_records = []
    for record in records:
        # 使用正则表达式提取字段
        match = re.match(r'(\d+),\s*"([^"]+)","([^"]+)",(.*),(\d+)', record)
        if match:
            id_num, entity, entity_type, description, rank = match.groups()
            # 清理描述中的换行和多余空格
            description = re.sub(r'\s+', ' ', description).strip()
            parsed_records.append([id_num, entity, entity_type, description, rank])
    
    # 写入处理后的数据
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入标题行
        writer.writerow(['id', 'entity', 'type', 'description', 'rank'])
        writer.writerows(parsed_records)
    
    print(f"处理完成，已将结果保存到 {output_file}")
    print(f"共处理 {len(parsed_records)} 条记录")

# 使用示例
if __name__ == "__main__":
    input_file = r"g:\Users\Kevin\Desktop\LightRAG-ui\entities_raw.txt"
    output_file = r"g:\Users\Kevin\Desktop\LightRAG-ui\entities_processed.csv"
    process_entities_file(input_file, output_file)