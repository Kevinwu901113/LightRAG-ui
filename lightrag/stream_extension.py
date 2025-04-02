import asyncio
from typing import AsyncGenerator, Dict, Any
from lightrag import LightRAG, QueryParam

def _generate_prompt(self, query: str, context: str, param: QueryParam = None) -> str:
    """
    生成提示文本
    
    Args:
        query: 用户查询
        context: 上下文信息
        param: 查询参数
        
    Returns:
        生成的提示文本
    """
    # 使用与 aquery 相同的提示模板
    if hasattr(self, 'prompt_template'):
        # 如果有自定义提示模板，使用它
        prompt = self.prompt_template.format(query=query, context=context)
    else:
        # 否则使用默认提示模板
        prompt = f"""请根据以下信息回答问题。如果无法从中得到答案，请说明无法回答。

信息：
{context}

问题：{query}

回答："""
    
    return prompt

async def astream(self, query: str, param: QueryParam = None) -> AsyncGenerator[str, None]:
    """
    流式生成回答
    
    Args:
        query: 用户查询
        param: 查询参数
        
    Yields:
        生成的文本片段
    """
    if param is None:
        param = QueryParam()
    
    # 获取上下文
    context = await self.aquery(query, param=QueryParam(mode=param.mode, only_need_context=True))
    
    # 构建提示
    prompt = self._generate_prompt(query, context, param)
    
    # 过滤掉不支持的参数
    model_kwargs = dict(self.llm_model_kwargs)
    
    # Ollama 不支持的参数列表
    unsupported_params = ['hashing_kv', 'cache_seed', 'cache_capacity']
    
    # 移除所有不支持的参数
    for param_name in unsupported_params:
        if param_name in model_kwargs:
            del model_kwargs[param_name]
    
    # 确保没有其他可能导致问题的参数
    if hasattr(param, '__dict__'):
        param_dict = param.__dict__.copy()  # 创建副本以避免修改原始对象
        for param_name in unsupported_params:
            if param_name in param_dict:
                param_dict.pop(param_name, None)
        
        # 使用清理后的参数字典
        clean_param = QueryParam()
        for k, v in param_dict.items():
            if k not in unsupported_params and hasattr(clean_param, k):
                setattr(clean_param, k, v)
        param = clean_param
    
    try:
        # 使用LLM模型流式生成，确保不传递不支持的参数
        result = self.llm_model_func(prompt, stream=True, **model_kwargs)
        
        # 如果是协程，先await它获取异步迭代器
        if asyncio.iscoroutine(result):
            async_iterator = await result
            async for token in async_iterator:
                yield token
        else:
            # 如果已经是异步迭代器，直接使用
            async for token in result:
                yield token
    except Exception as e:
        # 捕获并处理异常
        error_message = f"生成回答时出错: {str(e)}"
        yield error_message
        # 记录详细错误信息
        import traceback
        print(f"Error in astream: {str(e)}")
        print(traceback.format_exc())

# 将方法添加到LightRAG类
LightRAG._generate_prompt = _generate_prompt
LightRAG.astream = astream