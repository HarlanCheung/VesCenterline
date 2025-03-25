import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设你已经将 centerline 数据存储在 centerline.npy 文件中，或直接以 numpy 数组形式加载
centerline = np.load('/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/dsresults/centerline.npy')  # 或者直接使用你已有的 centerline 数组

# centerline 数组形状为 (N, 3)，通常存储的顺序为 (z, y, x)
# 在绘图时我们可以将 x, y, z 分别作为横、纵、深度坐标
x = centerline[:, 2]
y = centerline[:, 1]
z = centerline[:, 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 使用 scatter 展示中心线点，s=1 控制点的大小，你也可以调整为其他数值
ax.scatter(x, y, z, s=1, c='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Centerline Visualization')

plt.show()