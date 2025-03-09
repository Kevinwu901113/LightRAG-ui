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
import plotly.graph_objects as go
import numpy as np

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

def create_3d_graph_visualization(G):
    """
    使用Plotly创建3D交互式图形可视化
    """
    # 创建3D弹簧布局
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.5)
    
    # 提取节点位置
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]
    
    # 提取边位置
    x_edges = []
    y_edges = []
    z_edges = []
    
    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])
    
    # 基于节点度生成颜色
    node_colors = [G.degree(node) for node in G.nodes()]
    node_colors = np.array(node_colors)
    if node_colors.max() != node_colors.min():  # 避免除以零
        node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
    
    # 创建边的轨迹
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='lightgray', width=0.5),
        hoverinfo='none'
    )
    
    # 创建节点的轨迹
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(
            size=7,
            color=node_colors,
            colorscale='Viridis',
            colorbar=dict(
                title='节点度',
                thickness=10,
                x=1.1,
                tickvals=[0, 1],
                ticktext=['低', '高']
            ),
            line=dict(width=1)
        ),
        text=[node for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color='black'),
        hoverinfo='text'
    )
    
    # 创建3D图
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # 更新布局以获得更好的可视化效果
    fig.update_layout(
        title='3D知识图谱可视化',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=[
            dict(
                showarrow=False,
                text="交互式3D知识图谱可视化",
                xref="paper",
                yref="paper",
                x=0,
                y=0
            )
        ]
    )
    
    return fig

# 在 create_3d_graph_visualization 函数后添加一个新函数
def save_network_with_utf8(network, filename):
    """使用 UTF-8 编码保存网络图的 HTML 文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(network.html)

# 添加一个新函数来获取HTML内容
def get_html_content(file_path):
    """读取HTML文件并返回内容，处理可能的编码问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        with open(file_path, 'r', encoding='latin1') as f:
            return f.read()

# 页面布局
st.title("知识图谱可视化")

# 创建选项卡
tab1, tab2 = st.tabs(["2D可视化", "3D可视化"])

col_query, col_button = st.columns([8, 1])
with col_query:
    visual = st.text_input(
            "请输入想要生成知识图谱可视化所在的参考知识库路径（或留空使用默认路径）:",
            value=WORKING_DIR
        )
with col_button:
    run_button = st.button("🚀", help="运行知识图谱可视化")

# 现在run_button已定义，可以继续使用
if run_button:
    tem = visual
    file_path = f'{tem}/graph_chunk_entity_relation.graphml'  
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        G = nx.read_graphml(file_path)
        st.success(f"成功加载图，包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边")
        
        with tab1:
            st.subheader("2D网络图")
            # 2D可视化 - 使用pyvis
            net = Network(
                height="750px", 
                width="100%",
                notebook=False,
                cdn_resources="in_line",
                bgcolor="#ffffff",
                font_color="black",
                directed=False,  # 设置为无向图
                select_menu=True,  # 添加选择菜单
                filter_menu=True   # 添加过滤菜单
            )
            net.from_nx(G)
            
            # 添加物理布局参数，使节点更分散
            net.barnes_hut(
                gravity=-80000,  # 负值使节点相互排斥
                central_gravity=0.3,
                spring_length=250,  # 增加弹簧长度
                spring_strength=0.001,
                damping=0.09,
                overlap=0
            )
            
            # 增强节点可见性
            for node in net.nodes:
                node["size"] = 25  # 稍微减小节点大小
                node["color"] = "#4287f5"  # 添加明显的蓝色
                node["borderWidth"] = 2
                node["borderWidthSelected"] = 4
                node["font"] = {"size": 16, "face": "Arial", "color": "black"}
                node["label"] = node["id"]  # 确保节点有标签
                node["shape"] = "dot"  # 使用圆点形状
            
            for edge in net.edges:
                edge["width"] = 2
                edge["color"] = "#808080"
                edge["arrows"] = "to"  # 添加箭头
                edge["smooth"] = {"type": "continuous"}  # 平滑边
        
            # 生成并保存临时HTML文件
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)  # 确保目录存在
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = os.path.join(temp_dir, f"graph_{current_time}.html")  # 使用路径拼接
            
            # 保存并验证文件
            try:
                # 使用自定义的UTF-8保存函数，而不是net.save_graph
                save_network_with_utf8(net, temp_file)
                st.success(f"文件已保存到：{os.path.abspath(temp_file)}")
                
                # 验证文件存在
                if not os.path.exists(temp_file):
                    raise FileNotFoundError(f"{temp_file} 未生成")
                    
            except Exception as e:
                st.error(f"保存文件失败: {str(e)}")
                import traceback
                st.error(f"详细错误: {traceback.format_exc()}")
                
            # 在Streamlit中显示
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=800)
            except Exception as e:
                st.error(f"显示失败: {str(e)}")
                import traceback
                st.error(f"详细错误: {traceback.format_exc()}")
                
                # 尝试使用iframe作为备选方案
                st.markdown(f"""
                <iframe src="file://{os.path.abspath(temp_file)}" width="100%" height="800px"></iframe>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("3D交互式图")
            # 3D可视化 - 使用Plotly
            try:
                fig = create_3d_graph_visualization(G)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"3D可视化失败: {str(e)}")
                
    except Exception as e:
        st.error(f"加载图形文件失败: {str(e)}")