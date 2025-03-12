import json
import re
import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_embedding, hf_model_complete, siliconcloud_llm_response
from lightrag.utils import EmbeddingFunc
import json
from query import direct_query
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig

# 本地模型路径
# model_name_or_path = '/media/sata3/yjj/models/Qwen2.5-14B-Instruct'
model_name_or_path = '/media/sata3/yjj/models/glm-4-9b-chat'
# model_name_or_path = '/media/sata3/yjj/models/chatglm3-6b'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用 GPU 0 和 GPU 1

import asyncio
from threading import Thread
from functools import wraps, partial

def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""
    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


hf_model_complete = limit_async_func_call(1)(
    partial(
        hf_model_complete,
        # hashing_kv=llm_response_cache,
        # **llm_model_kwargs,
    )
)

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        print("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


# LightRAG 初始化部分
def init_rag():

    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['no_proxy'] = '127.0.0.1,localhost'
    
    WORKING_DIR = "./Docling"
    
    # logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=model_name_or_path,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8096,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "/media/sata3/yjj/models/bge-m3/",
                    cache_dir=r'/media/sata3/yjj/models',
                    # load_in_8bit=True,  # 使用 INT8 量化
                    # load_in_4bit=True,  # 使用 4bit 量化
                    device_map="auto",
                    model_max_length=8096
                ),
                embed_model=AutoModel.from_pretrained(
                    "/media/sata3/yjj/models/bge-m3/",
                    cache_dir=r'/media/sata3/yjj/models',
                    # load_in_8bit=True,  # 使用 INT8 量化
                    # load_in_4bit=True,  # 使用 4bit 量化
                    device_map="auto",
                ),
            ),
        ),
    )
    
    return rag


''' AutoRAG实现 '''

# 完整实现生成推理函数
async def async_generate_reasoning(query, rag_response,model_name='deepseek-chat') -> str:
    # reasoning_template = """[system]
    #     你是一个强大的问题规划专家，通过检索外部知识库来完成实现用户需求的问题检索方案规划。重点关注对话历史中是否有能够回答原始问题的信息。如果缺少必要信息，则规划如何进一步查询缺失信息。
    #
    #     [user]
    #     用户原始问题：{query}
    #     对话历史：{history}
    #
    #     """

    reasoning_template_system = """
    作为问题解决架构师，你负责通过系统性知识检索规划实现以下核心目标：

    【信息完整性验证体系】
    1. 已有信息总结：首先简单总结知识库回答。然后验证知识库响应是否完整覆盖原始问题的核心要素，对于已经存在的核心要素进行简单总结。可以从下面的可选维度开展分析：
       - 时间维度：是否需新增时间限定条件
       - 空间维度：是否需要拓展地域范围
       - 人物维度：是否缺乏特定人物信息
       - 数据维度：是否缺乏特定实体的某方面的信息
    2. 缺失信息维度：当检测到必要信息或核心要素缺失时，指出需要补充的缺失信息维度。
    
    ############################################
    【示例】
    ############################################
    示例1
    
    用户原始问题：
    分析1998-2020年长江中游水患对该地区的具体影响
    
    知识库回答：
    2003年长江中游发生特大洪水冲毁堤坝150米，造成直接经济损失8000万元
    ############################################
    Output:
    已有信息总结：
    2003年，长江中游发生特大洪水，冲毁堤坝150米，导致直接经济损失达8000万元。
       - 时间维度：2003年发生特大洪灾。
       - 数据维度：特大洪灾冲毁堤坝150米，造成直接经济损失8000万元
       - 空间维度：特大洪灾发生在长江中游
    
    缺失信息维度：
       - 缺少不同年份的长江中游水患信息
       - 具体的受影响行政区域
    ############################################
    示例2
    
    用户原始问题：
    元代修建的京杭大运河山东段在明清时期经历过哪些重要改造？
    
    知识库回答：
    《明史·河渠志》记载：永乐九年（1411年）工部尚书宋礼主持改造，采用汶上老人白英的"水柜"方案，在南旺湖设置分水枢纽，使河水流向南北的比例达到七分北注、三分南流。
    ############################################
    Output:
    已有信息总结：
    《明史·河渠志》记载，1411年，工部尚书宋礼采纳汶上老人白英的“水柜”方案，在南旺湖建立分水枢纽，成功调控河水，使其七分北流、三分南流。
       - 时间维度：永乐九年（1411年）采用了"水柜"改造方案。
       - 数据维度："水柜"改造方案使河水流向南北的比例达到七分北注、三分南流。
       - 空间维度："水柜"改造方案的分水枢纽设置在南旺湖
       - 人物维度："水柜"改造方案由汶上老人白英提出
    
    缺失信息维度：
       - 清代京杭大运河改造工程 
       - 清代京杭大运河山东段
       - 京杭大运河山东段重要改造
    ############################################
    """


    reasoning_template = """
    用户原始问题：
    {query}
    
    知识库回答：
    {rag_response}
    """
    
    formatted_prompt = reasoning_template.format(
        query=query,
        rag_response=rag_response
    )
    response= await direct_query(query=formatted_prompt,model_name=model_name,temperature=0.8,system_p=reasoning_template_system) 
    

    # response = await siliconcloud_llm_response(system_prompt=reasoning_template_system, history=[],
    #                                            content=formatted_prompt,model_name="deepseek-ai/DeepSeek-R1")

    logging.info(response)
    with open('log.txt', 'a', encoding='utf-8') as file:
        file.write("-------------reasoning---------------\n")
        file.write(response)

    # # todo：测试使用R1的思维链做reason
    # model_name_or_path: str = '/media/sata3/yjj/models/DeepSeek-R1-Distill-Llama-8B'
    #
    # # 使用与decompose_question相同的模型配置
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name_or_path,
    #     trust_remote_code=True,
    #     load_in_8bit=True,
    #     cache_dir='/media/sata3/yjj/models'
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name_or_path,
    #     trust_remote_code=True,
    #     load_in_8bit=True,
    #     device_map="auto",
    #     cache_dir='/media/sata3/yjj/models'
    # ).eval()
    
    # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(
    #     **inputs,
    #     generation_config=generation_config
    # )
    # outputs = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return response


def generate_reasoning(query, rag_response,model_name='deepseek-chat'):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(async_generate_reasoning(query, rag_response,model_name))

'''看RAG回答版本'''
# 判断节点
async def async_judge_node(query, rag_response, reasoning=None,modelname='deepseek-chat') -> str:
    judge_template_system = """
    你是一个聪明的AI助手，负责判断是否需要重写问题查询，遵循以下知识问答判定流程来输出。
    

    # 处理准则
    
    现在有两种输出方式：Final Answer和Refined Query供你选择，你必须选择其中一个。请你自行按照条件判断选择哪种输出。
    你的输出必须严格遵循以下两种输出格式，不得添加额外内容。
    
    ## 方式1. Final Answer
    
    ### 触发条件：
       - 基于可信的外部知识
       - 完整回答原始问题
       - 解决用户的所有问题
       - 保持专业正式语气
       
    ### 输出格式：
    Final Answer：正式回答内容，使用完整段落
        
    ## 方式2. Refined Query
    
    ### 触发条件：
       - 当信息存在以下情况时触发：
         * 检索结果不相关
         * 关键信息缺失
         * 存在歧义需要澄清
       - 必须明确标注需要补充的信息维度
       - 不能和历史中过去的优化查询相同
       - 可以参照查询规划思路来优化原始问题
       
    ### 输出格式：
    Refined Query：优化后的查询关键词。


    # 格式示例
    
    下面是一些基于上述知识问答判定流程进行回答的例子，用###分隔。
    注意，你的输出只能以Refined Query或者Final Answer开头，绝对不能包含如Intermediate Answer或者Retrieved Document之类的开头。
    
    ###
    
    Question：基于"差序格局"理论，分析长三角农村彩礼金额与村民社交网络中心度的关联性。

    Retrieved Document_1：费孝通《乡土中国》提出差序格局概念，认为传统农村社会关系以个人为中心向外推及血缘、地缘构成波纹状差序。
    
    Intermediate Answer_1：理论框架建立完成。需实证数据分析。
    
    Refined Query：浙江农村彩礼调查GIS可视化数据
    
    Retrieved Document_2：浙江大学2022年对湖州10个行政村的研究显示：家族网络中心度排名前20%的家庭，平均彩礼金额38.2万元，比尾部20%家庭高出127%。
    
    Intermediate Answer_2：数据印证中心度与彩礼正相关。需机制阐释。
    
    Refined Query：差序格局与婚姻资源交换模型
    
    Retrieved Document_3：社会学家边燕杰的嵌入性理论指出：高网络中心度家族通过高额彩礼增强联姻对象家族的义务绑定，这种现象在宗族文化保留度＞60%的村落更显著（相关系数0.72）。
    
    Final Answer：在差序格局作用下，核心家族通过高额彩礼：
    
    强化与联姻家族的义务纽带
    巩固自身在亲属网络中的枢纽地位
    形成每10万元彩礼增加0.3个社交节点的非线性关系
    
    ###
    
    Question：元代修建的京杭大运河山东段在明清时期经历过哪些重要改造？
    
    Retrieved Document_1：元代至元二十六年（1289年）开挖会通河，实现漕船全程直达。但由于水位落差问题，济宁到临清段常淤塞，漕运效率低下。
    
    Intermediate Answer_1：元代初始工程存在航道运维问题。需确认明清维护政策。
    
    Refined Query：明朝永乐朝京杭运河山东段改造
    
    Retrieved Document_2：《明史·河渠志》记载：永乐九年（1411年）工部尚书宋礼主持改造，采用汶上老人白英的"水柜"方案，在南旺湖设置分水枢纽，使河水流向南北的比例达到七分北注、三分南流。
    
    Intermediate Answer_2：明代建立南旺分水枢纽。继续搜索清代工程。
    
    Refined Query：清代康熙朝运河山东段治理
    
    Retrieved Document_3：康熙四十二年（1703年），总河张鹏翮在戴村坝增筑石工，建成三合土坝体，使汶水入运比例提升至70%。同期开挖陶城铺新河，缩短航道80里。
    
    Final Answer：重要改造包括：
    
    1411年明永乐年间：建造南旺分水枢纽
    1703年清康熙时期：加固戴村坝并改道陶城铺
    1730年清雍正朝：建立十里闸水位调控系统
    
    ###
    
    Question：电影《黄土高坡》的导演去世于哪一年？

    Retrieved Document_1：《黄土高坡》是由中国大陆导演王建军执导的1988年农村题材影片，该片获得第12届百花奖最佳影片。影片通过西北农村家庭三代人的故事，展现改革开放初期的社会变迁。原著改编自作家李怀沙的同名小说。
    
    Intermediate Answer_1：根据Retrieved Document_1，《黄土高坡》的导演是王建军。但此文档未提及导演的去世时间。要回答问题，需获取王建军去世的具体年份。
    
    Refined Query:导演王建军 生平
    
    Retrieved Document_2：王建军（1935年3月18日－2019年5月3日），陕西榆林人，中国第四代导演代表人物。早年在北京电影学院任教，后转型导演，代表作品包括《山乡巨变》《大河长歌》。1998年获中国电影导演终身成就奖。晚年致力于纪录片《非遗中国》制作。
    
    Intermediate Answer_2：根据Retrieved Document_2，导演王建军于2019年5月3日逝世。故《黄土高坡》导演去世年份为2019年。
    
    Final Answer：2019年。
    
    ###
    
    Question：黄鹤楼与昌德宫是否处于同一历史时期建造？
    
    Retrieved Document_1：黄鹤楼始建于三国吴黄武二年（公元223年），最初作为军事瞭望台建造，历代屡毁屡建。现存建筑为1985年重建，采用钢筋混凝土框架仿木结构，高度51.4米，比清代旧楼高出近20米。
    
    Intermediate Answer_1：根据Retrieved Document_1，黄鹤楼始建时间为223年。但无昌德宫相关信息。需要确认昌德宫建造时间。
    
    Refined Query:韩国昌德宫建造年代
    
    Retrieved Document_2：昌德宫（창덕궁）是韩国首尔的宫殿群，正宫景福宫被烧毁后，昌德宫作为朝鲜王朝主要宫殿使用。1405年（朝鲜太宗五年）开始营建，历经多次火灾，现存宫殿多为1609年光海君时期重建，被誉为最具传统风格的宫殿建筑群。
    
    Intermediate Answer_2：昌德宫始建于1405年，而黄鹤楼创始于223年。两者建造时间相差1182年，不在同一历史时期（三国时期与高丽王朝时期）。因此答案是否定的。
    
    Final Answer：否。
    
    ###
    
    Question：《白鹿原》电影导演与《饮食男女》导演是否拥有相同教育背景？
    
    Retrieved Document_1：2012年电影《白鹿原》由第五代导演王全安执导，该片改编自陈忠实同名小说，耗时9年筹备，获柏林电影节艺术贡献银熊奖。影片因叙事结构与篇幅限制引发原著读者争议。
    
    Intermediate Answer_1：根据Retrieved Document_1，《白鹿原》导演为王全安。但未提及教育背景信息。需查询王全安的学历背景。
    
    Refined Query:王全安学历
    
    Retrieved Document_2：王全安，1965年生于延安，1977年考取北京舞蹈学院芭蕾舞专业，后转入北京电影学院导演系1987级，班主任为谢飞导演。是张艺谋、陈凯歌之后中国第六代导演代表。
    
    Intermediate Answer_2：王全安毕业于北京电影学院导演系。现需确认《饮食男女》导演李安的教育背景。
    
    Refined Query:李安学历
    
    Retrieved Document_3：李安，1954年生，1975年毕业于台湾艺术专科学校（今台艺大）影剧科，1979年进入伊诺大学香槟分校戏剧系获得学士学位，1982年于纽约大学Tisch艺术学院获得电影制作硕士学位。《推手》《喜宴》等片获国际认可。
    
    Intermediate Answer_3：李安先后就读台湾艺专、美国纽约大学电影系。王全安就读北京电影学院。两者均接受电影高等教育但院校不同，王全安接受中国传统电影教育体系，李安具有国际院校背景。因此教育背景不完全相同。
    
    Final Answer：否。
    
    ###
    
    """

    judge_template_v1 = """
    请你处理以下用户输入：
    
    Question：
    {query}
    
    Intermediate Answer：
    {reasoning}
    
    Retrieved Document：
    {rag_response}
    """

    judge_template_v2 = """
    请你处理以下用户输入：
    
    Question：
    {query}
    
    Retrieved Document：
    {rag_response}
    """

    if reasoning == None:
        formatted_prompt = judge_template_v2.format(
            query=query,
            rag_response=rag_response
        )
    else:
        formatted_prompt = judge_template_v1.format(
            query=query,
            rag_response=rag_response,
            reasoning=reasoning
        )

    # print("开始调用API")
    # # deepseek-v3的API调用
    # response = await siliconcloud_llm_response(system_prompt=judge_template_system, history=[], content=formatted_prompt)
    # print("完成回复")

    # 本地模型调用
    response= await direct_query(query=formatted_prompt,model_name=modelname,temperature=0.8,system_p=judge_template_system)

    return response



def judge_node(query, rag_response, reasoning=None,modelname='deepseek-chat'):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(async_judge_node(query, rag_response, reasoning,modelname))


# 主流程
def test_thread(question, rag, history=None):
    # 初始化迭代参数
    max_iter = 5
    current_iter = 0
    final_answer = None
    
    # 初始检索
    initial_tem = {"ans": "", "source": []}
    logging.info(f"\n=== 初始问题处理 ===")
    ans, sources = rag.query(question, param=QueryParam(mode="hybrid"))
    initial_tem["ans"] = ans

    # # 不总结
    # initial_tem["source"] = sources

    # 总结节点
    for source_ in sources:
        ans1 = rag.query1(ans, source_, question)
        initial_tem["source"].append(ans1)
    
    # 构建对话历史
    if history is None:
        history = []
    history.append({
        "type": "initial",
        "query": question,
        "answer":ans,
        # "response": initial_tem
        "sources": sources
    })
    
    # 首次判断
    logging.info(f"\n=== 初始判定 ===")
    judge_output = judge_node(
        query=question,
        rag_response=json.dumps(initial_tem, ensure_ascii=False)
    )

    logging.info(f"判断输出：{judge_output}")

    with open('log.txt', 'a', encoding='utf-8') as file:
        file.write("\n-------------初始问题处理---------------\n")
        file.write(str(history))
        file.write("\n-------------初始判定---------------\n")
        file.write(judge_output)

    # 处理循环
    while max_iter > 0 and current_iter <= max_iter:
        # print(f"\n=== 第 {current_iter + 1} 轮迭代 ===")
        logging.info((f"\n=== 第 {current_iter + 1} 轮迭代 ==="))
        with open('log.txt', 'a', encoding='utf-8') as file:
            file.write(f"\n=== 第 {current_iter + 1} 轮迭代 ===\n")

        if 'Final Answer：' in judge_output:
            logging.info(judge_output)
            final_answer = judge_output.split("Final Answer：")[1].strip()
            # final_answer = judge_output
            break

        elif 'Refined Query' in judge_output:
            refined_query = judge_output.split("Refined Query：")[1].strip()

            # 不查询分解，直接召回
            tem = {"ans": "", "source": []}
            logging.info(f"\n=== 初始问题处理 ===")
            ans, sources = rag.query(refined_query, param=QueryParam(mode="hybrid"))
            tem["ans"] = ans

            # # 不总结
            # tem["source"] = sources

            # 总结
            for source_ in sources:
                ans1 = rag.query1(ans, source_, question)
                tem["source"].append(ans1)

            # 生成推理规划
            reasoning = generate_reasoning(
                query=refined_query,
                rag_response=json.dumps(tem, ensure_ascii=False)
            )

            # print("推理节点输出: ", reasoning)
            logging.info(f"推理节点输出: {reasoning}")

            # 更新判断 带reasoning
            judge_output = judge_node(
                query=question,  # 始终基于原始问题判断
                rag_response=json.dumps(tem, ensure_ascii=False),
                reasoning=reasoning
            )

            history.append({
                "type": "refined",
                "query": refined_query,
                "reasoning": reasoning,
                "response": {
                    "ans": ans,
                    "sources": tem["source"]
                }
            })

            # print("判断节点输出：", judge_output)
            logging.info(f"判断节点输出: {judge_output}")

            with open('log.txt', 'a', encoding='utf-8') as file:
                file.write(f"\n判断节点输出: {judge_output}\n")

            current_iter += 1

        max_iter -= 1

    # 结果处理
    if final_answer:
        logging.info(f"\n=== Final Answer ===")
        # print(final_answer)
        logging.info(final_answer)
        logging.info(f"历史记录：{history}")

        with open('log.txt', 'a', encoding='utf-8') as file:
            file.write(f"\n=== Final Answer ===\n")
            file.write(final_answer)
            file.write(f"\n历史记录：{str(history)}\n")

        return True, {
        "ans": final_answer,
        # "source": sources,
        "history": history
        }
        # return {
        #     "status": "success",
        #     "answer": final_answer,
        #     "iterations": current_iter,
        #     "history": history
        # }
    else:
        # print(f"\n=== 未找到Final Answer ===")
        logging.info(f"\n=== 未找到Final Answer ===")
        logging.info(f"历史记录：{history}")

        with open('log.txt', 'a', encoding='utf-8') as file:
            file.write(f"\n=== 未找到Final Answer ===\n")
            file.write(f"\n历史记录：{str(history)}\n")

        return False, {
        "ans": initial_tem["ans"],
        # "source": sources,
        "history": history
        }
        # return {
        #     "status": "max_iter_reached",
        #     "answer": initial_tem["ans"],  # 返回初始答案
        #     "iterations": current_iter,
        #     "history": history
        # }


def RAG_response(question, rag):
    tem = {"ans": "", "source": []}

    ans, sources = rag.query(question, param=QueryParam(mode="hybrid"))

    tem["ans"] = ans
    for source_ in sources:
        ans1 = rag.query1(ans, source_, question)
        tem["source"].append(ans1)
    return tem

if __name__ == "__main__":

    logging.basicConfig(
        filename='AutoRAG.log',  # 日志文件名
        level=logging.INFO,  # 记录INFO及以上级别的日志
        format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 定义日期时间格式
    )

    rag = init_rag()

    all_question = [
        "2021年，玉塘街道化解“国满件”几宗，在光明政府在线主动公开政府信息几条？",
        "2021年，马田街道街道级河长每旬巡河域次数，一年累计巡河巡域次数？",
        "李松蓢水系和干流上游水系的流域面积分别为多少平方千米？",
        "公明镇棉纱统销税和存款利息所得税的税率分别为？",
    ]
    for question in all_question:
        _, result = test_thread(
        question=question,
        rag=rag,
        history=[]
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

    import torch.distributed as dist
    dist.destroy_process_group()
