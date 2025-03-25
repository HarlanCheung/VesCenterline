import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, binary_erosion, maximum_filter, generate_binary_structure
from skimage.morphology import skeletonize_3d
import heapq
import matplotlib.pyplot as plt

def load_image(path):
    """加载三维tif图像"""
    img = tifffile.imread(path)
    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")
    print(f"实际像素值范围: {np.min(img)}-{np.max(img)}")
    print(f"唯一像素值: {np.unique(img)}")
    return img.astype(np.uint8)

def validate_binary(img):
    """将任意二值格式标准化为0-1"""
    unique = np.unique(img)
    
    if len(unique) == 2:
        # 如果是其他二值格式（如0-255）
        normalized = np.where(img > 0, 1, 0).astype(np.uint8)
        print(f"检测到二值图像，值域 {unique} 已标准化为0-1")
        return normalized
    else:
        raise ValueError(f"非二值图像！检测到值域：{unique}")

def show_slice(img, title):
    """显示中间切片"""
    plt.imshow(img[img.shape[0]//2], cmap='gray')
    plt.title(title)
    plt.show()

def compute_boundary(binary_volume):
    """计算三维边界"""
    # 使用3x3x3结构元素
    struct = np.ones((3,3,3), dtype=bool)
    eroded = binary_erosion(binary_volume, structure=struct)
    boundary = binary_volume ^ eroded  # XOR操作获取边界
    
    # 验证边界计算
    assert np.sum(boundary) > 0, "边界检测失败！"
    print(f"边界体素数: {np.sum(boundary)}")
    return boundary

def compute_dt(binary_volume):
    """计算欧氏距离变换"""
    dt = distance_transform_edt(binary_volume)
    
    # 基础验证
    assert np.max(dt) > 0, "距离变换异常：最大距离为0"
    print(f"距离范围: {np.min(dt):.2f}-{np.max(dt):.2f}")
    return dt

def find_local_maxima_adaptive(dt, min_size=3, max_size=15):
    """
    使用自适应窗口大小的局部最大值检测，利用窗口大小离散性矢量化计算。
    参数：
      - dt: 3D 距离变换图，值域为 0-18
      - min_size: 最小窗口大小
      - max_size: 最大窗口大小
    返回：
      - candidates: 3D 二值数组，标记局部最大值位置
    """
    # 计算每个点的窗口大小：线性映射 dt 到 [min_size, max_size]
    local_window_size = (dt / 18) * (max_size - min_size) + min_size  
    local_window_size = np.clip(local_window_size, min_size, max_size).astype(int)
    
    # 存储候选点的布尔数组
    candidates = np.zeros_like(dt, dtype=bool)
    
    # 取所有唯一的窗口尺寸
    unique_sizes = np.unique(local_window_size)
    
    # 对于每个唯一窗口尺寸，进行一次全局的最大滤波，再选出满足条件的体素
    for s in unique_sizes:
        # 构造当前窗口尺寸的掩码，标记 dt 中窗口大小为 s 的位置
        mask = (local_window_size == s)
        if np.any(mask):
            # 构造立方体结构元素（全1的布尔数组）
            footprint = np.ones((s, s, s), dtype=bool)
            # 对整个 dt 进行最大滤波
            filtered = maximum_filter(dt, footprint=footprint)
            # 对于 mask 中的点，如果当前点值等于滤波后的值，则认为是局部最大值
            candidates[mask] = dt[mask] == filtered[mask]
    
    print(f"自适应局部最大值候选点数: {np.sum(candidates)}")
    return candidates

def fast_marching_centerline(candidates, dt, connectivity=2):
    """使用 Fast Marching 方法细化和连接中心线"""
    
    # 初始化中心线
    centerline = np.zeros_like(candidates, dtype=bool)
    
    # 获取所有候选点坐标
    candidate_coords = np.argwhere(candidates)
    
    # 建立最小优先队列（Heap Queue）
    pq = []
    
    # 记录访问状态
    visited = np.zeros_like(dt, dtype=bool)
    
    # 初始化 Fast Marching，优先传播 dt 值较大的点
    for z, y, x in candidate_coords:
        heapq.heappush(pq, (-dt[z, y, x], z, y, x))  # 注意这里是 -dt，保证最大 dt 先被访问
    
    # 定义 26 连接邻域
    neighbors_offset = np.array([
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
        [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
        [0, -1, -1], [0, -1, 0], [0, -1, 1],
        [0, 0, -1], [0, 0, 1],
        [0, 1, -1], [0, 1, 0], [0, 1, 1],
        [1, -1, -1], [1, -1, 0], [1, -1, 1],
        [1, 0, -1], [1, 0, 0], [1, 0, 1],
        [1, 1, -1], [1, 1, 0], [1, 1, 1]
    ])

    # Fast Marching 过程
    while pq:
        _, z, y, x = heapq.heappop(pq)  # 取出当前 dt 最高的点
        
        if visited[z, y, x]:  # 如果已经访问，跳过
            continue

        visited[z, y, x] = True  # 标记为已访问
        centerline[z, y, x] = True  # 设为中心线

        # 遍历邻居
        for dz, dy, dx in neighbors_offset:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < dt.shape[0] and 0 <= ny < dt.shape[1] and 0 <= nx < dt.shape[2]:
                if not visited[nz, ny, nx]:
                    heapq.heappush(pq, (-dt[nz, ny, nx], nz, ny, nx))  # 继续传播
    
    return centerline
# def fast_marching_centerline(candidates, dt, connectivity=2):
#     """
#     使用NumPy优化的快速行进法，用于细化和连接中心线
    
#     参数:
#     candidates: 候选中心线点 (bool array)
#     dt: 距离变换图像
#     connectivity: 连接性 (1: 6-邻域, 2: 18-邻域, 3: 26-邻域)
    
#     返回:
#     细化后的中心线 (bool array)
#     """
#     # 初始化中心线
#     centerline = candidates.copy()
    
#     # 定义邻域
#     if connectivity == 1:  # 6-邻域
#         neighbors_offset = np.array([[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[-1,0,0],[1,0,0]])
#     elif connectivity == 2:  # 18-邻域
#         neighbors_offset = np.array([
#             [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
#             [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
#             [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
#             [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1]
#         ])
#     else:  # 26-邻域
#         neighbors_offset = np.array([
#             [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
#             [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
#             [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
#             [0, -1, -1], [0, -1, 0], [0, -1, 1],
#             [0, 0, -1], [0, 0, 1],
#             [0, 1, -1], [0, 1, 0], [0, 1, 1],
#             [1, -1, -1], [1, -1, 0], [1, -1, 1],
#             [1, 0, -1], [1, 0, 0], [1, 0, 1],
#             [1, 1, -1], [1, 1, 0], [1, 1, 1]
#         ])
    
#     # 获取候选点的坐标
#     candidate_coords = np.argwhere(candidates)
    
#     # 遍历每个候选点
#     for z, y, x in candidate_coords:
#         # 找到当前点的邻居
#         neighbor_coords = np.array([
#             [z + dz, y + dy, x + dx]
#             for dz, dy, dx in neighbors_offset
#         ])
        
#         # 移除超出图像边界的邻居
#         valid_neighbors = []
#         for nz, ny, nx in neighbor_coords:
#             if (0 <= nz < candidates.shape[0] and
#                 0 <= ny < candidates.shape[1] and
#                 0 <= nx < candidates.shape[2]):
#                 valid_neighbors.append([nz, ny, nx])
        
#         valid_neighbors = np.array(valid_neighbors)
        
#         # 如果没有邻居，则跳过
#         if len(valid_neighbors) == 0:
#             continue
        
#         # 计算邻居的距离变换值
#         neighbor_values = dt[valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]]
        
#         # 如果邻居的距离变换值都小于当前点，则保留当前点
#         if np.all(neighbor_values < dt[z, y, x]):
#             continue
        
#         # 否则，移除当前点
#         centerline[z, y, x] = False
    
#     return centerline

def postprocess(centerline):
    """三维骨架化"""
    return skeletonize_3d(centerline)

def save_result(result, path):
    tifffile.imsave(path, result.astype(np.uint8))
    print(f"结果已保存至 {path}")

def main(input_path):
    # 1. 加载数据
    img = load_image(input_path)
    # 2. 标准化验证
    binary_img = validate_binary(img)  # 修改此处
    #show_slice(binary_img, "Binary image")  # 修改此处
    
    # 3.计算边界
    boundary = compute_boundary(binary_img)  # 修改此处
    #show_slice(boundary, "Boundary voxels")
    
    # 4. 计算距离变换
    dt = compute_dt(binary_img)  # 修改此处
    #show_slice(dt, "Distance map")
    
    # 5. 局部极大值
    candidates = find_local_maxima_adaptive(dt)
    #show_slice(candidates, "Candidates")
    
    # 6. 中心线提取
    centerline = fast_marching_centerline(candidates, dt)
    print(f"中心线体素数: {np.sum(centerline)}")
    # 7. 后处理
    final = postprocess(centerline)
    print(f"最终中心线体素数: {np.sum(final)}")
    
    # 保存结果
    output_path1 = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/centerline_thin.tif'
    output_path2 = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/centerline_raw.tif'
    tifffile.imwrite(output_path1, final.astype(np.uint8) * 255)  # 转换为 0/255 的图像
    tifffile.imwrite(output_path2, centerline.astype(np.uint8) * 255)  # 转换为 0/255 的图像
    print(f"中心线已保存至: {output_path2}")
    print(f"最终中心线已保存至: {output_path1}")

if __name__ == "__main__":
    main("/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif")