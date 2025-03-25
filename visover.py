import imageio
import numpy as np
import matplotlib.pyplot as plt

# 读取TIFF stack，第200张图片索引为199 (0-based index)
img1_stack = imageio.mimread('/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif')
img2_stack = imageio.mimread('/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/skraw.tif')



assert len(img1_stack) >= 200 and len(img2_stack) >= 200, "Stack 少于200张切片"

img1 = img1_stack[177].astype(np.float32)
img2 = img2_stack[177].astype(np.float32)

# 归一化到 0-1
img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())

# 设置阈值，用于区分叠加区域
threshold = 0.05  # 可自行调整

# 创建RGB图像
rgb_img = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.float32)

# 红色通道来自 img1
rgb_img[..., 0] = img1_norm

# 蓝色通道来自 img2
rgb_img[..., 2] = img2_norm

# 对蓝色图像亮度高于threshold的区域，强制让红色通道归零，避免紫色出现
mask_blue = img2_norm > threshold
rgb_img[mask_blue, 0] = 0  # 红色通道置零

# 显示结果
plt.figure(figsize=(8,8))
plt.imshow(rgb_img)
plt.axis('off')
plt.title('Slice 178: Red=bin_image, Blue=skeleton')
plt.show()