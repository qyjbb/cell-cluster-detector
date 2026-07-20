# 安装 pytorch (根据你的显卡配置，建议去pytorch官网找对应版本，这里是CPU/通用版)
# pip install torch torchvision
#
# 安装核心算法和可视化库
# pip install cellpose streamlit opencv-python-headless matplotlib

# --- 🔴 关键系统补丁 (必须放在最前面) ---
import os

# 允许 OpenMP 库重复加载（解决 Windows DLL 冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # 强制让 PyTorch 第一个加载
# ---------------------------------------

import streamlit as st
import numpy as np
import cv2
from cellpose import models
from PIL import Image

# --- 页面配置 ---
st.set_page_config(page_title="聚团细胞检测 (v3核心版)", layout="wide")

st.title("🔬 聚团细胞自动检测与框选系统 (v3适配版)")
st.markdown("**后台模型**: Cellpose v3 (Model-Based) | **数据集兼容**: BBBC006 / 显微镜图像")

# --- 侧边栏：参数设置 ---
st.sidebar.header("参数配置")

# CPU模式强制锁定
use_gpu = st.sidebar.checkbox("使用 GPU 加速", value=False, disabled=True, help="CPU环境已锁定")

model_type = st.sidebar.selectbox(
    "选择模型类型",
    ('cyto', 'nuclei', 'cyto3'),  # 增加了 cyto3，这是v3版本的强力模型
    index=1,
    help="'nuclei' 适合 BBBC006 (细胞核); 'cyto3' 是最新版通用模型"
)

# CellposeModel 需要明确的直径，或者设为 None 让它自己猜（但比较慢）
diameter = st.sidebar.number_input("细胞估算直径 (像素)", value=30, min_value=0, help="BBBC006 约为 30-40。设为0则自动估算(较慢)")


# --- 核心功能函数 ---
@st.cache_resource
def load_model(type_model, use_gpu):
    # 🌟 核心修改：直接使用 CellposeModel，这是 v3 版本最稳健的类
    # 你的日志显示 'CellposeModel' 是存在的，所以我们用它
    return models.CellposeModel(gpu=use_gpu, model_type=type_model)


def plot_results(image, masks):
    """
    根据 Mask 绘制边界框
    """
    img_out = image.copy()

    # 转RGB以便画彩框
    if len(img_out.shape) == 2:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)

    n_cells = masks.max()

    # 遍历每一个检测到的细胞
    for i in range(1, n_cells + 1):
        y, x = np.where(masks == i)

        if len(y) > 0 and len(x) > 0:
            top, bottom = np.min(y), np.max(y)
            left, right = np.min(x), np.max(x)

            # 画红色矩形框
            cv2.rectangle(img_out, (left, top), (right, bottom), (255, 0, 0), 2)

    return img_out, n_cells


# --- 主逻辑 ---

# 1. 加载模型
with st.spinner(f"正在初始化 Cellpose v3 ({model_type})..."):
    try:
        model = load_model(model_type, use_gpu)
    except Exception as e:
        st.error(f"模型加载崩溃: {e}")
        st.stop()

# 2. 文件上传
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # 简单的通道处理
    if len(img_np.shape) > 2 and img_np.shape[2] >= 3:
        # 如果是彩色图，转灰度给模型看通常更准（针对nuclei），或者保持原样
        # 为了兼容性，我们这里取一个通道或者转灰度
        # 但 CellposeModel 也能吃 RGB。这里为了 BBBC006 (通常是单通道) 做个兜底
        if model_type == 'nuclei':
            # 简单的转灰度策略
            img_input = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        else:
            img_input = img_np
    else:
        img_input = img_np

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图像")
        st.image(img_np, use_column_width=True)

    if st.button("开始检测 (Run Detection)"):
        with st.spinner("正在推理 (Inference)..."):
            try:
                # 🌟 核心修改：适配 CellposeModel 的 eval 接口
                # v3 的 CellposeModel.eval 返回 3 个值: masks, flows, styles
                # (原来的 Cellpose 类返回 4 个，多一个 diams，这里我们删掉了接收 diams)

                # 预处理直径参数
                diam_arg = diameter if diameter > 0 else None
                channels = [0, 0]  # 灰度图模式

                masks, flows, styles = model.eval(
                    img_input,
                    diameter=diam_arg,
                    channels=channels
                )

                # 绘制
                result_img, cell_count = plot_results(img_np, masks)

                with col2:
                    st.subheader(f"检测结果 (计数: {cell_count})")
                    st.image(result_img, use_column_width=True)

                st.success(f"成功检测到 {cell_count} 个目标！")

                with st.expander("查看分割 Mask"):
                    st.image(masks, clamp=True, use_column_width=True)

            except Exception as e:
                st.error(f"推理过程报错: {e}")
                st.info("💡 提示：如果是尺寸不匹配错误，请尝试调整'细胞直径'参数")