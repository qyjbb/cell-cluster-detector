import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import streamlit as st
from cellpose import models
from PIL import Image


st.set_page_config(
    page_title="聚团细胞检测可视化平台",
    layout="wide",
)

st.title("聚团细胞检测可视化平台")
st.markdown("基于 Cellpose 的细胞图像分割与目标框可视化工具。")


st.sidebar.header("检测参数")

use_gpu = st.sidebar.checkbox(
    "使用 GPU",
    value=False,
    disabled=True,
    help="当前版本默认使用 CPU，便于在普通电脑上运行。",
)

model_type = st.sidebar.selectbox(
    "模型类型",
    ("cyto", "nuclei", "cyto3"),
    index=1,
    help="nuclei 适合细胞核图像，cyto/cyto3 适合更通用的细胞图像。",
)

diameter = st.sidebar.number_input(
    "细胞估算直径（像素）",
    value=30,
    min_value=0,
    help="设置为 0 时由模型自动估算，手动设置通常速度更快。",
)


@st.cache_resource
def load_model(model_name: str, gpu: bool):
    return models.CellposeModel(gpu=gpu, model_type=model_name)


def prepare_image(image_array: np.ndarray, model_name: str) -> np.ndarray:
    if image_array.ndim == 3 and model_name == "nuclei":
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return image_array


def draw_bounding_boxes(image_array: np.ndarray, masks: np.ndarray):
    result = image_array.copy()

    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    cell_count = int(masks.max())

    for label in range(1, cell_count + 1):
        y_coords, x_coords = np.where(masks == label)

        if len(y_coords) == 0 or len(x_coords) == 0:
            continue

        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))

        cv2.rectangle(result, (left, top), (right, bottom), (255, 0, 0), 2)

    return result, cell_count


with st.spinner(f"正在加载模型：{model_type}"):
    try:
        model = load_model(model_type, use_gpu)
    except Exception as exc:
        st.error(f"模型加载失败：{exc}")
        st.stop()


uploaded_file = st.file_uploader(
    "上传细胞图像",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("请上传一张细胞图像开始检测。")
    st.stop()


image = Image.open(uploaded_file)
image_array = np.array(image)
model_input = prepare_image(image_array, model_type)

source_column, result_column = st.columns(2)

with source_column:
    st.subheader("原始图像")
    st.image(image_array, use_column_width=True)

if st.button("开始检测"):
    with st.spinner("正在检测细胞区域..."):
        try:
            target_diameter = diameter if diameter > 0 else None
            channels = [0, 0]

            masks, flows, styles = model.eval(
                model_input,
                diameter=target_diameter,
                channels=channels,
            )

            result_image, cell_count = draw_bounding_boxes(image_array, masks)

            with result_column:
                st.subheader(f"检测结果：{cell_count} 个目标")
                st.image(result_image, use_column_width=True)

            st.success(f"检测完成，共识别 {cell_count} 个细胞目标。")

            with st.expander("查看分割 Mask"):
                st.image(masks, clamp=True, use_column_width=True)

        except Exception as exc:
            st.error(f"检测失败：{exc}")
            st.info("可以尝试调整细胞直径参数，或更换模型类型后重新检测。")
