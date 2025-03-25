import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile as tiff

def load_image(input_path):
    """加载图像并转换为二值图像"""
    bin_img = tiff.imread(input_path)
    print("图像的形状是:", bin_img.shape)
    bin_img = (bin_img > 0).astype(np.uint8)  # 转换为二值图像
    return bin_img

def DT(bin_img):
    """计算距离变换"""
    distance_map = distance_transform_edt(bin_img)
    return distance_map

def is_simple_point(temp_img, z, y, x):
    """
    判断体素 (z, y, x) 是否为简单点
    通过检查26邻域的连通性来近似判断
    """
    # 获取26邻域
    neighborhood = temp_img[max(0, z-1):min(z+2, temp_img.shape[0]),
                            max(0, y-1):min(y+2, temp_img.shape[1]),
                            max(0, x-1):min(x+2, temp_img.shape[2])].copy()
    
    # 删除当前体素
    dz, dy, dx = [s - max(0, t-1) for s, t in zip([z, y, x], [z, y, x])]
    temp_neighborhood = neighborhood.copy()
    temp_neighborhood[dz, dy, dx] = 0
    
    # 检查连通分量
    from scipy.ndimage import label
    labeled_before, num_before = label(neighborhood)
    labeled_after, num_after = label(temp_neighborhood)
    
    # 如果删除后连通分量数量不变，则为简单点
    return num_before == num_after

def collapse_centerline(distance_map, bin_img):
    """塌陷过程，提取中心线"""
    z, y, x = bin_img.shape
    centerline = np.zeros_like(bin_img, dtype=bool)  # 中心线标记数组
    temp_img = bin_img.copy()  # 复制图像用于修改

    # 获取所有前景体素的坐标和距离值
    foreground_voxels = np.argwhere(temp_img == 1)
    distances = distance_map[temp_img == 1]
    # 按距离值升序排序
    sorted_indices = np.argsort(distances)
    sorted_voxels = foreground_voxels[sorted_indices]

    # 塌陷过程
    for voxel in sorted_voxels:
        z_idx, y_idx, x_idx = voxel
        if temp_img[z_idx, y_idx, x_idx] == 1:  # 确保体素仍在前景中
            if is_simple_point(temp_img, z_idx, y_idx, x_idx):
                # 如果是简单点，删除
                temp_img[z_idx, y_idx, x_idx] = 0
            else:
                # 如果不是简单点，标记为中心线
                centerline[z_idx, y_idx, x_idx] = True

    return centerline

# 主程序
input_path = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif'
bin_img = load_image(input_path)
distance_map = DT(bin_img)
centerline = collapse_centerline(distance_map, bin_img)
print("中心线体素数量:", np.sum(centerline))

# 在主程序后添加导出部分
output_path = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/centerline_output.tif'

# 将中心线保存为 TIFF 文件
tiff.imwrite(output_path, centerline.astype(np.uint8) * 255)  # 转换为 0/255 的图像
print(f"中心线已保存至: {output_path}")