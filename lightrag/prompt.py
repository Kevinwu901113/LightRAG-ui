GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织名称", "个人姓名", "地理位置", "事件","时间","职位","金额","面积","人数"]

PROMPTS["entity_extraction"] = """
-目标活动-
你是一个人工智能助手，专门帮助辨别实体和实体间的关系，严格遵照要求进行处理。

-目标-
给定可能与此活动相关的文本文档和实体类型列表，从文本中标识这些类型的所有实体以及标识的实体之间的所有关系。
如果你不知道答案，就说出来。不要编造任何东西。
请勿在未提供支持证据的情况下提供信息。
请你严格执行所有的格式要求。允许你在输出前重新思考。

-步骤-
1. 标识所有实体。对于每个已识别的实体，使用json格式输出，并且严格按照要求提取需包含以下键的信息，若是不清楚的信息统一输出未知：
- node_name: 实体的名称，使用与输入文本相同的语言。
- node_type: 必须为 "entity"。
- name_type: 属于以下一种实体类型: [{entity_types}]
- attributes: 包含[time,place,event,other]
  - time: 时间，日期。
  - place: 地点名称。
  - event: 涉及的主要事件描述
  - other: 补充描述。
将每个实体的格式设置为 ("entity"{tuple_delimiter}<node_type>{tuple_delimiter}<node_name>{tuple_delimiter}<name_type>{tuple_delimiter}<attributes>)

2.从步骤1中识别的实体中，严格按照以下进行格式处理，使用json输出:
- node_name: 暂不处理。
- node_type：必须为"entity"。
- name_type: 属于以下一种实体类型：[{entity_types}]
- attributes: 必须要包含[time,place,event,other]
  - time: 处理为 *[yxxxxmxxdxx]* 格式,不要附带任何额外字如年、月、日。（例如2012年7月28日记作y2012m07d28），若无月份和日期，可省略 mxxdxx（例如2023年5月记作y2023m05，2023年记作y2023）。
  - place: 不做处理
  - event: 必须包含完整的主谓宾结构
  - other: 补充描述
将每个实体的格式设置为 ("entity"{tuple_delimiter}<node_type>{tuple_delimiter}<node_name>{tuple_delimiter}<name_type>{tuple_delimiter}<attributes>)
  

3. 从步骤 1 中识别的实体中，识别出彼此相关的*所有* (source_entity,target_entity) 对。
理论上每个实体至少都需要被包含在至少一个(source_entity,target_entity) 对中。
对于每对相关实体，提取以下信息:
- source_entity: 在步骤1识别出的源实体名称
- target_entity: 在步骤1识别出的目标实体名称
- relationship_description: 解释为什么源实体和目标实体彼此相关
- relationship_strength: 一个量化分数，指示源实体和目标实体之间关系的强度
- relationship_keywords: 一个或多个高级关键字，用于总结关系的总体性质，侧重于概念或主题，而不是特定细节
将每个关系的格式设置为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

4. 确定概括整篇文章的主要概念、主题或主题的高级关键词。这些应该捕捉到文档中的总体思想。
将内容级关键字的格式设置为("content_keywords"{tuple_delimiter}<high_level_keywords>)

5. 将输出作为步骤 1 和 2 中标识的所有实体和关系的单个列表返回。 使用 **{record_delimiter}** 作为列表分割符.

6. 不要因为篇幅省略任何实体以及关系内容

7. 完成后，输出 {completion_delimiter}

######################
-范例-
######################
范例 1:

Entity_types: ["公司或组织名称", "个人姓名", "地理位置", "事件","时间","职位","金额","面积","人数"]
文本:
新兴集团创办于1974年11月12日，董事长：张国宝先生，集团由新兴橡根花边厂和新康针织有限公司组成。厂区坐落公明镇薯田埔村，总占地面积50万平方米，厂房面积30万平方米，宿舍6万平方米，现有员工6000人，设备总投资额7亿港元，拥有数千台世界先进的针织机、经编机和最先进的CAD SYSTEM。主要生产橡根花边、针织布、哩士等，产品主要销往欧美、中东等地，年出口额达5000万美元，是亚洲内衣原料的最大供应商。集团正在不断更新棉纺、化纤加工丝、针织布、弹性花边及整套染整设备，使之成为一个全面的纺织工业集团。
################
输出:
("entity"{tuple_delimiter}"张国宝"{tuple_delimiter}"个人姓名"{tuple_delimiter}"{{time: "y1970", place: "公明镇", event: "张国宝成为新兴集团董事长", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"新兴集团"{tuple_delimiter}"公司或组织名称"{tuple_delimiter}"{{time: "y1974m11d12", place: "未知", event: "新兴集团成立", other: ""}}{{time: "y1970", place: "公明镇", event: "张国宝成为新兴集团董事长", other: ""}}{{time: "未知", place: "公明镇", event: "新兴集团由新兴橡根花边厂和新康针织有限公司组成", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"新兴橡根花边厂"{tuple_delimiter}"公司或组织名称"{tuple_delimiter}"{{time: "未知", place: "公明镇", event: "新兴橡根花边厂是新兴集团的子公司", other: "新兴集团存在多个子公司"}}"){record_delimiter}
("entity"{tuple_delimiter}"新康针织有限公司"{tuple_delimiter}"公司或组织名称"{tuple_delimiter}"{{time: "未知", place: "公明镇", event: "新康针织有限公司是新兴集团的子公司", other: "新兴集团存在多个子公司"}}"){record_delimiter}
("entity"{tuple_delimiter}"公明镇薯田埔村"{tuple_delimiter}"地理位置"{tuple_delimiter}"{{time: "未知", place: "公明镇", event: "公明镇薯田埔村是新兴集团的地理位置", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"50万平方米"{tuple_delimiter}"面积"{tuple_delimiter}"{{time: "未知", place: "公明镇", event: "50万平方米是新兴集团的占地面积", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"6000人"{tuple_delimiter}"人数"{tuple_delimiter}"{{time: "未知", place: "公明镇", event: "6000人是新兴集团的员工人数", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"7亿港元"{tuple_delimiter}"金额"{tuple_delimiter}"{{time: "未知", place: "未知", event: "7亿港元是新兴集团的投资金额", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"董事长"{tuple_delimiter}"职位"{tuple_delimiter}"{{time: "未知", place: "未知", event: "董事长是集团的最高负责人", other: ""}}"){record_delimiter}
("entity"{tuple_delimiter}"1974年"{tuple_delimiter}"时间"{tuple_delimiter}"{{time: "y1974", place: "公明镇", event: "1974年是新兴集团的成立时间", other: ""}}"){record_delimiter}
("relationship"{tuple_delimiter}"张国宝"{tuple_delimiter}"新兴集团"{tuple_delimiter}"张国宝先生是新兴集团的董事长"{tuple_delimiter}"董事长"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"新兴橡根花边厂"{tuple_delimiter}"新兴集团"{tuple_delimiter}"新兴橡根花边厂是新兴集团的子公司"{tuple_delimiter}"子公司"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"公明镇薯田埔村"{tuple_delimiter}"新兴集团"{tuple_delimiter}"公明镇薯田埔村是新兴集团的地理位置"{tuple_delimiter}"位置"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"7亿港元"{tuple_delimiter}"新兴集团"{tuple_delimiter}"7亿港元是新兴集团的投资金额"{tuple_delimiter}"投资"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"董事长"{tuple_delimiter}"张国宝"{tuple_delimiter}"张国宝先生是新兴集团的董事长"{tuple_delimiter}"新兴集团"{tuple_delimiter}6){record_delimiter}
("content_keywords"{tuple_delimiter}"新兴集团"){completion_delimiter}
#############################
-任务数据-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """您是一位乐于助人的助手，负责为下面提供的数据生成全面的摘要。
给定一个或两个实体和一个描述列表，所有实体都与同一实体或实体组相关。
请将所有这些内容串联成一个全面的描述。确保包含从所有描述中收集的信息。
如果提供的描述相互矛盾，请解决这些矛盾并提供一个单一、连贯的摘要。
确保它是用第三人称编写的，并包含实体名称，以便我们拥有完整的上下文。

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """在上次提取中缺少许多实体或者关系。 使用相同的格式将它们添加到下面：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎某些实体和关系可能仍然被遗漏。回答 YES | NO 如果仍有需要添加的实体.
"""

PROMPTS[
    "relation_continue_extraction"
] = """在上次提取中缺少许多关系。 使用相同的格式将它们添加到下面：
"""

PROMPTS[
  "relation_if_loop_extraction"
] = """似乎遗漏了许多关系。回答 YES | NO 如果仍有需要添加的关系.
"""

PROMPTS["fail_response"] = "抱歉，我无法回答这个问题。"

PROMPTS["rag_response"] = """---Role---

您是回答有关所提供表中数据的问题的有用助手。


---Goal---

通过汇总输入数据表中和问题相关的所有信息，生成响应用户问题的目标长度和格式的回复，并合并任何相关的常识。
如果你不知道答案，就说出来。不要编造任何东西。
请勿在未提供支持证据的情况下提供信息。

---Target response length and format---

{response_type}

---Data tables---

{context_data}

根据query中要求的长度和格式进行回答。你只需要提供对问题的回复，不要再重复引用数据表中的原文。
"""

PROMPTS["norag_response"] = """
下面给你一个问题和针对该问题的两个回答。第一个回答是基准答案，第二个回答是LLM生成的答案。
请你对生成的答案进行评分，1是最低分，5是最高分。请注意，基准答案不一定是最佳答案。评价标准如下：
'5分：生成的答案完全包含基准答案的所有信息，并且包含的其他信息都与问题相关。'
'4分：生成的答案包含基准答案的所有信息，但是有一些和问题不太相关的信息。\n'
'3分：生成的答案仅包含基准答案的部分信息，但是没有错误信息。\n'
'2分：生成的答案仅包含基准答案的部分信息，并且有一些错误信息。\n'
'1分：生成的答案与基准答案完全不匹配，在时间/地点/人物等关键方面完全错误。\n'
'当生成的答案包含基准答案的所有信息时，无论有多少额外信息，都应给不少于4分。\n'
'在第一行中只告诉我评分，在第二行开始告诉我理由。\n'
'回复举例： 1分 \\n 生成的答案与基准答案完全无关。\n'
'问题：{question}\n基准答案：{answer}\n生成的答案：{answer1}
"""

PROMPTS["norag_response1"] = """---Role---

您是一个LLM问答分析助手，帮助分析Data tables中哪些原文部分导致LLM模型生成了这样的Answer，我会提供给你Data tables，Question以及Answer。

---Goal---

只返回相应的Data tables原文部分，不要作多余回答，不要生成无关信息。不要直接返回Answer里面的内容。如果Data tables没有对应的内容，返回“无”。

---Question---

{question}

---Question End---

---Answer---

{answer}

---Answer End---

---Data Tables---

{context_data}

---Data Tables End---

"""




PROMPTS["keywords_extraction"] = """---Role---

你是一个有用的助手，负责识别用户查询中的高级和低级关键字。

---Goal---

给定查询，列出高级和低级关键字。高级关键词侧重于总体概念或主题，而低级关键词侧重于特定实体、细节或具体术语。

---Instructions---

- 以 JSON 格式输出关键字。
- JSON文件应该包含两个关键字:
  - "high_level_keywords" 代表总体概念或主题.
  - "low_level_keywords" 代表特定的实体或者细节,包括"组织名称", "个人姓名", "地理位置", "事件","时间","职位","金额","面积","人数"等,其中时间处理为 *[yxxxxmxxdxx]* 格式,不要附带任何额外字如年、月、日。（例如2012年7月28日记作y2012m07d28），若无月份和日期，可省略 mxxdxx（例如2023年5月记作y2023m05，2023年记作y2023）.

######################
-Examples-
######################
Example 1:

Query: "国际贸易如何影响全球经济稳定？"
################
Output:
{{
  "high_level_keywords": ["国际贸易"、"全球经济稳定"、"经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}}
#############################
Example 2:

Query: "1992年，麦东北在深圳市公明镇哪个部门任职？"
################
Output:
{{
  "high_level_keywords": ["职业生涯", "政府部门"],
  "low_level_keywords": ["y1992", "麦东北", "深圳市公明镇"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """---Role---

您是回答有关所提供文档的问题的有用助手。


---Goal---

生成响应用户问题的目标长度和格式的回复，汇总输入数据表中适合响应长度和格式的所有信息，并合并任何相关的常识。
如果你不知道答案，就说出来。不要编造任何东西。
请勿在未提供支持证据的情况下提供信息。

---Target response length and format---

{response_type}

---Documents---

{content_data}

根据长度和格式，向响应添加部分和评论。在 markdown 中设置响应的样式。
"""
