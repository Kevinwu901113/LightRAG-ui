def process_entities_file(input_file, output_file):
    """
    使用状态机方法处理包含不规则换行的实体文件
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过文件开头的空行和代码块标记
    start_idx = 0
    for i, line in enumerate(lines):
        if 'id,' in line and 'entity,' in line and 'type,' in line:
            start_idx = i + 1
            break
    
    processed_lines = []
    current_record = ""
    in_quotes = False
    
    for line in lines[start_idx:]:
        # 检查是否在引号内
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
        
        # 如果这一行以数字开头且不在引号内，说明是新记录的开始
        if re.match(r'^\d+,', line.strip()) and not in_quotes:
            if current_record:
                processed_lines.append(current_record)
            current_record = line.strip()
        else:
            # 否则，将这一行添加到当前记录
            current_record += " " + line.strip()
    
    # 添加最后一条记录
    if current_record:
        processed_lines.append(current_record)
    
    # 写入处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题行
        f.write('id,entity,type,description,rank\n')
        for line in processed_lines:
            # 清理多余的空格
            line = re.sub(r'\s+', ' ', line)
            f.write(line + '\n')
    
    print(f"处理完成，已将结果保存到 {output_file}")
    print(f"共处理 {len(processed_lines)} 条记录")