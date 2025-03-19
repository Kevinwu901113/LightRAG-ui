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
import threading
from queue import Queue

# 默认工作目录
WORKING_DIR = "./knowledge_base"

# ===== GraphML文件处理函数 =====
def read_graphml_file(file_path):
    """直接使用networkx读取GraphML文件，并显示进度条"""
    try:
        # 添加进度条
        progress_bar = st.progress(0)
        st.info("正在加载图形文件...")
        
        # 模拟加载进度
        for i in range(101):
            # 更新进度条
            progress_bar.progress(i)
            if i < 30:
                # 文件读取阶段
                if i == 20:
                    G = nx.read_graphml(file_path)
            elif i < 70:
                # 图形处理阶段
                pass
            else:
                # 完成阶段
                pass
            
            # 控制进度条速度
            if i < 90:
                import time
                time.sleep(0.01)  # 适当的延迟，使进度条可见
        
        # 完成后清除进度条
        progress_bar.empty()
        
        return G
    except Exception as e:
        # 出错时也清除进度条
        if 'progress_bar' in locals():
            progress_bar.empty()
        raise Exception(f"读取GraphML文件失败: {str(e)}")

def read_graphml(file_path):  
    """读取GraphML文件并返回根元素"""
    tree = ET.parse(file_path)  
    root = tree.getroot()  
    return root 

def extract_nodes_edges(root):  
    """从XML根元素提取节点和边"""
    nodes = []  
    edges = []  
    for child in root:  
        if child.tag == 'node':  
            nodes.append(child)  
        elif child.tag == 'edge':  
            edges.append(child)  
    return nodes, edges 

def parse_attributes(element):  
    """解析元素的属性"""
    attributes = {}  
    for attr in element:  
        if attr.tag == 'data':  
            key = attr.get('key')  
            attributes[key] = attr.get('value')  
    return attributes 

def build_graph(nodes, edges):  
    """从节点和边构建图"""
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

# ===== 可视化函数 =====
def create_3d_graph_visualization(G):
    """使用Plotly创建3D交互式图形可视化"""
    # 创建3D弹簧布局
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.5)
    
    # 提取节点位置
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]
    
    # 提取边位置
    x_edges, y_edges, z_edges = [], [], []
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
            xaxis=dict(showbackground=True, backgroundcolor='white'),
            yaxis=dict(showbackground=True, backgroundcolor='white'),
            zaxis=dict(showbackground=True, backgroundcolor='white'),
            bgcolor='white'
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
        ],
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_2d_network_visualization(G):
    """创建2D网络可视化"""
    net = Network(
        height="750px", 
        width="100%",
        notebook=False,
        cdn_resources="in_line",
        bgcolor="#ffffff",
        font_color="black",
        directed=False,
        select_menu=True,
        filter_menu=True
    )
    
    # 从networkx图转换
    net.from_nx(G)
    
    # 添加物理布局参数，使节点更分散
    net.barnes_hut(
        gravity=-80000,
        central_gravity=0.3,
        spring_length=250,
        spring_strength=0.001,
        damping=0.09,
        overlap=0
    )
    
    # 增强节点可见性
    for node in net.nodes:
        node["size"] = 25
        node["color"] = "#4287f5"
        node["borderWidth"] = 2
        node["borderWidthSelected"] = 4
        node["font"] = {"size": 16, "face": "Arial", "color": "black"}
        node["label"] = node["id"]
        node["shape"] = "dot"
    
    for edge in net.edges:
        edge["width"] = 2
        edge["color"] = "#808080"
        edge["arrows"] = "to"
        edge["smooth"] = {"type": "continuous"}
    
    # 清理可能包含问题字符的数据
    # for node in net.nodes:
    #     if "title" in node and node["title"]:
    #         # 移除或替换可能导致编码问题的字符
    #         node["title"] = str(node["title"]).encode('ascii', 'ignore').decode('ascii')
    #     if "label" in node and node["label"]:
    #         # 确保标签是ASCII兼容的
    #         node["label"] = str(node["label"]).encode('ascii', 'ignore').decode('ascii')
    
    return net

# ===== 文件处理函数 =====
def save_network_with_utf8(network, filename):
    """使用UTF-8编码保存网络图的HTML文件"""
    try:
        # 确保网络图已经生成HTML
        if not hasattr(network, 'html') or not network.html:
            network.write_html(filename)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(network.html)
        
        # 验证文件是否成功写入且不为空
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            return True
        else:
            raise Exception("保存的HTML文件为空")
    except Exception as e:
        import traceback
        st.error(f"保存网络图失败: {str(e)}")
        st.error(f"详细错误: {traceback.format_exc()}")
        return False

def get_html_content(file_path):
    """读取HTML文件并返回内容，处理可能的编码问题"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"文件为空: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"文件内容为空: {file_path}")
            return content
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"文件内容为空: {file_path}")
            return content
    except Exception as e:
        import traceback
        st.error(f"读取HTML文件失败: {str(e)}")
        st.error(f"详细错误: {traceback.format_exc()}")
        raise

def generate_temp_file_path():
    """生成临时文件路径"""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(temp_dir, f"graph_{current_time}.html")

# ===== 主应用函数 =====
def prepare_2d_visualization(G):
    """准备2D可视化数据，不直接显示"""
    # 创建网络
    net = create_2d_network_visualization(G)
    
    # 生成并保存临时HTML文件
    temp_file = generate_temp_file_path()
    
    # 保存文件
    try:
        # 确保网络图已经准备好
        if not hasattr(net, 'html') or not net.html:
            # 强制生成HTML
            net.generate_html()
        
        # 不使用pyvis的write_html方法，而是直接使用UTF-8编码写入文件
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(net.html)
        
        # 验证文件是否成功写入
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            raise Exception("保存的HTML文件为空或不存在")
        
        return {"status": "success", "file_path": temp_file, "abs_path": os.path.abspath(temp_file)}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

def prepare_3d_visualization(G):
    """准备3D可视化数据，不直接显示"""
    try:
        fig = create_3d_graph_visualization(G)
        return {"status": "success", "figure": fig}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

def display_2d_result(result):
    """在主线程中显示2D可视化结果"""
    st.subheader("2D网络图")
    
    if result["status"] == "error":
        st.error(f"2D可视化准备失败: {result['error']}")
        st.error(f"详细错误: {result['traceback']}")
        return
    
    temp_file = result["file_path"]
    
    # 检查文件是否存在且不为空
    if not os.path.exists(temp_file):
        st.error(f"文件不存在: {temp_file}")
        return
    
    if os.path.getsize(temp_file) == 0:
        st.error(f"生成的HTML文件为空: {temp_file}")
        return
    
    st.success(f"文件已保存到：{result['abs_path']}")
    
    # 在Streamlit中显示
    try:
        html_content = get_html_content(temp_file)
        if not html_content or not html_content.strip():
            st.error("HTML内容为空")
            return
            
        # 显示HTML内容
        components.html(html_content, height=800, scrolling=True)
    except Exception as e:
        st.error(f"显示失败: {str(e)}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        
        # 尝试使用iframe作为备选方案
        st.warning("尝试使用备选方案显示...")
        try:
            # 使用相对URL而不是文件路径
            rel_path = os.path.relpath(temp_file, os.getcwd())
            st.markdown(f"""
            <iframe src="{rel_path}" width="100%" height="800px"></iframe>
            """, unsafe_allow_html=True)
        except Exception as iframe_error:
            st.error(f"备选方案也失败了: {str(iframe_error)}")

def display_3d_result(result):
    """在主线程中显示3D可视化结果"""
    st.subheader("3D交互式图")
    
    if result["status"] == "error":
        st.error(f"3D可视化准备失败: {result['error']}")
        st.error(f"详细错误: {result['traceback']}")
        return
    
    # 显示图形
    st.plotly_chart(result["figure"], use_container_width=True)

def main():
    """主函数"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 页面布局
    st.title("知识图谱可视化")
    
    # 创建输入区域
    col_query, col_button = st.columns([8, 1])
    with col_query:
        visual = st.text_input(
            "请输入想要生成知识图谱可视化所在的参考知识库路径（或留空使用默认路径）:",
            value=WORKING_DIR
        )
    with col_button:
        run_button = st.button("🚀", help="运行知识图谱可视化")
    
    # 初始化会话状态
    if "graph_loaded" not in st.session_state:
        st.session_state.graph_loaded = False
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = None
    if "result_2d" not in st.session_state:
        st.session_state.result_2d = None
    if "result_3d" not in st.session_state:
        st.session_state.result_3d = None
    
    # 创建选项卡
    tab1, tab2 = st.tabs(["2D可视化", "3D可视化"])
    
    # 处理按钮点击
    if run_button:
        file_path = f'{visual}/graph_chunk_entity_relation.graphml'
        
        try:
            # 加载图（现在包含进度条）
            G = read_graphml_file(file_path)
            st.success(f"成功加载图，包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边")
            
            # 保存图数据到会话状态
            st.session_state.graph_loaded = True
            st.session_state.graph_data = G
            
            # 生成两种可视化结果
            with st.spinner('正在生成2D可视化...'):
                st.session_state.result_2d = prepare_2d_visualization(G)
            with st.spinner('正在生成3D可视化...'):
                st.session_state.result_3d = prepare_3d_visualization(G)
                
        except Exception as e:
            st.error(f"加载图形文件失败: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
    
    # 在选项卡中显示结果
    with tab1:
        if st.session_state.result_2d:
            display_2d_result(st.session_state.result_2d)
    
    with tab2:
        if st.session_state.result_3d:
            display_3d_result(st.session_state.result_3d)

if __name__ == "__main__":
    main()