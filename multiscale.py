from skimage import io
from skimage.measure import label, regionprops
import scipy.ndimage as nd
import numpy as np

# 1. 读取 3D TIFF 堆栈
input_path = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif"
volume = io.imread(input_path)

# 将图像转换为布尔类型：非零值视为 True
bool_volume = volume > 0

# 2. 定义 4x4x4 的结构元素
struct_elem = np.ones((3, 3, 3), dtype=bool)


# 3. 进行两遍 4x4x4 腐蚀操作
# 第一次腐蚀
eroded_volume1 = nd.binary_erosion(bool_volume, structure=struct_elem)
# 第二次腐蚀
eroded_volume2 = nd.binary_erosion(eroded_volume1, structure=struct_elem)

# 4. 连通分量分析：移除孤立小球结构
# 对腐蚀后的图像进行标记
labels = label(eroded_volume2, connectivity=1)
cleaned_volume = eroded_volume2.copy()

# 定义最小连通体阈值（体素数量）
min_size = 100  # 根据实际情况调整

# 遍历每个连通区域，移除体素数量小于 min_size 的区域
for region in regionprops(labels):
    if region.area < min_size:
        cleaned_volume[labels == region.label] = False

# 5. 保存处理后的 TIFF 堆栈
output_path = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/thick_eroded_vessels_3d_33_2.tif"
# 将布尔类型转换为 uint8（False->0, True->255）后保存
io.imsave(output_path, (cleaned_volume.astype(np.uint8) * 255))