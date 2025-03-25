import tifffile as tiff
import numpy as np
from vedo import Volume, show

def visualize_surface_from_tiff(tiff_path):
    """读取 TIFF 文件并渲染为中心线的表面，添加光照效果"""
    # 读取 TIFF 文件
    data = tiff.imread(tiff_path)
    print("图像的形状是:", data.shape)
    print("数据的最小值:", np.min(data), "最大值:", np.max(data))
    
    # 检查数据是否有效
    if np.max(data) == 0:
        print("警告：数据中没有非零值，无法渲染！")
        return
    
    # 转换为 uint8 类型并二值化（确保 0 和 255）
    if data.dtype != 'uint8':
        data = (data > 0).astype('uint8') * 255
    
    # 创建体视对象
    vol = Volume(data)
    
    # 提取等值面（表面），直接传入阈值 127
    surface = vol.isosurface(127)  # 移除 'threshold=' 参数，直接传入值
    
    # 设置表面属性
    surface.color('red')          # 表面颜色为红色
    surface.alpha(1.0)            # 不透明度为 1
    surface.lighting(
        ambient=0.3,              # 环境光强度
        diffuse=0.7,              # 漫反射强度
        specular=0.2,             # 高光强度
        specular_power=20         # 高光锐度
    )
    
    # 显示表面渲染
    show(surface, axes=1, bg='black', size=(800, 800), interactive=True)

# 主程序
tiff_path = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/postsk.tif'  # 修改为您的 TIFF 文件路径
visualize_surface_from_tiff(tiff_path)