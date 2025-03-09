import xml.etree.ElementTree as ET
import networkx as nx 
import matplotlib.pyplot as plt
import streamlit as st
from pyvis.network import Network
import pandas as pd
import streamlit.components.v1 as components
import matplotlib
import os
from datetime import datetime
WORKING_DIR = "./dickens1"


def read_graphml(file_path):  
    tree = ET.parse(file_path)  
    root = tree.getroot()  
    return root 

def extract_nodes_edges(root):  
    nodes = []  
    edges = []  
    for child in root:  
        if child.tag == 'node':  
            nodes.append(child)  
        elif child.tag == 'edge':  
            edges.append(child)  
    return nodes, edges 

def parse_attributes(element):  
    attributes = {}  
    for attr in element:  
        if attr.tag == 'data':  
            key = attr.get('key')  
            attributes[key] = attr.get('value')  
    return attributes 

def build_graph(nodes, edges):  
    G = nx.Graph()  
    for node in nodes:  
        attributes = parse_attributes(node)  
        G.add_node(node.get('id'), **attributes)  
    for edge in edges:  
        source = edge.get('source')  
        target = edge.get('target')  
        attributes = parse_attributes(edge)  
        G.add_edge(source, target, **attributes)  
    return G  

# def got_func(physics):
#   got_net = Network(height="600px", width="100%", font_color="black",heading='Game of Thrones Graph')

# # set the physics layout of the network
#   got_net.barnes_hut()
#   got_data = pd.read_csv("https://www.macalester.edu/~abeverid/data/stormofswords.csv")
#   #got_data = pd.read_csv("stormofswords.csv")
#   #got_data.rename(index={0: "Source", 1: "Target", 2: "Weight"}) 
#   sources = got_data['Source']
#   targets = got_data['Target']
#   weights = got_data['Weight']

#   edge_data = zip(sources, targets, weights)

#   for e in edge_data:
#     src = e[0]
#     dst = e[1]
#     w = e[2]

#     got_net.add_node(src, src, title=src)
#     got_net.add_node(dst, dst, title=dst)
#     got_net.add_edge(src, dst, value=w)

#   neighbor_map = got_net.get_adj_list()

# # add neighbor data to node hover data
#   for node in got_net.nodes:
#     node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
#     node["value"] = len(neighbor_map[node["id"]])
#   if physics:
#     got_net.show_buttons(filter_=['physics'])
#   got_net.show("gameofthrones.html")
  

# def simple_func(physics): 
#   nx_graph = nx.cycle_graph(10)
#   nx_graph.nodes[1]['title'] = 'Number 1'
#   nx_graph.nodes[1]['group'] = 1
#   nx_graph.nodes[3]['title'] = 'I belong to a different group!'
#   nx_graph.nodes[3]['group'] = 10
#   nx_graph.add_node(20, size=20, title='couple', group=2)
#   nx_graph.add_node(21, size=15, title='couple', group=2)
#   nx_graph.add_edge(20, 21, weight=5)
#   nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)


#   nt = Network("500px", "500px",notebook=True,heading='')
#   nt.from_nx(nx_graph)
#   #physics=st.sidebar.checkbox('add physics interactivity?')
#   if physics:
#     nt.show_buttons(filter_=['physics'])
#   nt.show('test.html')


# def karate_func(physics): 
#   G = nx.karate_club_graph()


#   nt = Network("500px", "500px",notebook=True,heading='Zachary’s Karate Club graph')
#   nt.from_nx(G)
#   #physics=st.sidebar.checkbox('add physics interactivity?')
#   if physics:
#     nt.show_buttons(filter_=['physics'])
#   nt.show('karate.html')

col_query, col_button = st.columns([8, 1])
with col_query:
    visual=st.text_input(
            "请输入想要生成知识图谱可视化所在的参考知识库路径（或留空使用默认路径）:",
            value=WORKING_DIR
        )
with col_button:
    run_button = st.button("🚀", help="运行知识图谱可视化")
if run_button:
    
    tem=visual

    file_path = f'{tem}/graph_chunk_entity_relation.graphml'  
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文
    plt.rcParams['axes.unicode_minus'] = False
    print(file_path)
    G= nx.read_graphml(file_path)
    print(f"成功加载图，包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边")
    net = Network(
        height="750px", 
        width="100%",
        notebook=False,  # 必须设置为False
        cdn_resources="in_line"  # 关键参数：内联资源
    )
    net.from_nx(G)
    
    # 自定义节点配置
    for node in net.nodes:
        node["size"] = 30
        node["font"] = {"size": 14, "face": "SimHei"}
    
    for edge in net.edges:
        edge["width"] = 1.5
        edge["color"] = "#808080"

    # 生成并保存临时HTML文件
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # 确保目录存在
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = os.path.join(temp_dir, f"graph_{current_time}.html")  # 使用路径拼接
    
    # 保存并验证文件
    try:
        net.save_graph(temp_file)  # 改用save_graph
        print(f"文件已保存到：{os.path.abspath(temp_file)}")
        
        # 验证文件存在
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"{temp_file} 未生成")
            
    except Exception as e:
        st.error(f"保存文件失败: {str(e)}")
        

    # 在Streamlit中显示
    try:
        with open(temp_file, "r", encoding="utf-8") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=800)
    except Exception as e:
        st.error(f"显示失败: {str(e)}")
    # fig, ax = plt.subplots()
    # pos = nx.spring_layout(G)  # 使用 spring_layout
    # nx.draw_networkx_nodes(
    #     G, pos,
    #     node_size=800,
    #     node_color="skyblue",
    #     alpha=0.9,
    #     edgecolors="darkblue",
    #     linewidths=1.5
    # )
    
    # # 边样式设置
    # nx.draw_networkx_edges(
    #     G, pos,
    #     width=1.5,
    #     edge_color="gray",
    #     alpha=0.6
    # )
    
    # # 标签样式设置
    # nx.draw_networkx_labels(
    #     G, pos,
    #     font_size=10,
    #     font_family="SimHei",  # 确保与前面设置的字体一致
    #     font_color="darkblue",
    #     verticalalignment="bottom"
    # )
    # nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold')
    # plt.tight_layout()
    # st.pyplot(fig)
    

# st.sidebar.title('Choose your favorite Graph')
# option=st.sidebar.selectbox('select graph',('Simple','Karate', 'GOT'))
# physics=st.sidebar.checkbox('add physics interactivity?')
# simple_func(physics)

# if option=='Simple':
#   HtmlFile = open("test.html", 'r', encoding='utf-8')
#   source_code = HtmlFile.read() 
#   components.html(source_code, height = 900,width=900)


# got_func(physics)

# if option=='GOT':
#   HtmlFile = open("gameofthrones.html", 'r', encoding='utf-8')
#   source_code = HtmlFile.read() 
#   components.html(source_code, height = 1200,width=1000)



# karate_func(physics)

# if option=='Karate':
#   HtmlFile = open("karate.html", 'r', encoding='utf-8')
#   source_code = HtmlFile.read() 
#   components.html(source_code, height = 1200,width=1000)