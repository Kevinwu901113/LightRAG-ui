import ollama
import traceback
import subprocess
import json

def get_model_names():
    """使用ollama的API获取模型列表"""
    try:
        models = ollama.list()
        if models and 'models' in models and isinstance(models['models'], list):
            # 添加调试信息，查看模型数据结构
            print(f"模型数据结构: {models['models'][0] if models['models'] else '空列表'}")
            # 检查每个模型对象的结构，适应可能的不同键名
            model_names = []
            for model in models['models']:
                if 'name' in model:
                    model_names.append(model['name'])
                elif 'model' in model:
                    model_names.append(model['model'])
                else:
                    # 如果找不到预期的键，尝试获取第一个值作为模型名称
                    if model:
                        first_key = next(iter(model))
                        model_names.append(model[first_key])
            return model_names
        return []
    except Exception as e:
        print(f"获取模型列表失败: {str(e)}")
        # 添加更详细的错误信息
        print(f"详细错误: {traceback.format_exc()}")
        return []

def get_model_by_name(model_name):
    """根据名称获取模型"""
    try:
        models = ollama.list()
        if 'models' in models:
            for model in models['models']:
                if ('name' in model and model['name'] == model_name) or \
                   ('model' in model and model['model'] == model_name):
                    return model
        return None
    except Exception as e:
        print(f"获取模型信息失败: {str(e)}")
        return None

def get_model_info(model_name):
    """
    获取指定模型的详细信息
    
    Args:
        model_name (str): 模型名称
        
    Returns:
        dict: 模型信息字典
    """
    try:
        # 使用ollama API获取模型信息
        model_info = ollama.show(model_name)
        return model_info
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        return {}