import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, convolve
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itk
import heapq

# 数据加载
def load_tiff_stack(filepath):
    img = tifffile.imread(filepath)
    print(f"[DEBUG] Loaded image shape: {img.shape}, dtype: {img.dtype}")
    binary_img = (img > 0).astype(np.uint8)
    print(f"[DEBUG] Converted image to binary with unique values: {np.unique(binary_img)}")
    return binary_img

# 计算边界
def compute_3d_boundary(binary_stack):
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(binary_stack, struct)
    boundary = binary_stack ^ eroded
    print("[DEBUG] Computed 3D boundary.")
    return boundary

# 距离变换
def compute_3d_distance_transform(binary_stack):
    dt = distance_transform_edt(binary_stack)
    print("[DEBUG] Computed 3D distance transform. Stats - min: {:.2f}, max: {:.2f}".format(np.min(dt), np.max(dt)))
    return dt

# 前沿坍缩点检测
def detect_3d_collapsing_fronts(dt, r=2):
    dt_discrete = np.floor(dt).astype(int)
    max_d = np.max(dt_discrete)
    collapsing_points = np.zeros_like(dt, dtype=bool)
    kernel = np.ones((2*r+1, 2*r+1, 2*r+1), dtype=np.uint8)
    for d in tqdm(range(max_d, 0, -1), desc="Detecting collapsing fronts"):
        current_front = (dt_discrete == d)
        unvisited = (dt_discrete > d) | (dt_discrete == 0)
        has_unvisited = convolve(unvisited.astype(np.uint8), kernel, mode='constant', cval=0) > 0
        collapsing_d = current_front & (~has_unvisited)
        collapsing_points |= collapsing_d
    print("[DEBUG] Detected 3D collapsing fronts. Total collapsing points: {}".format(np.sum(collapsing_points)))
    return collapsing_points


def fast_marching_centerline_cpu(binary, dt, initial_seeds, tau=0.5):
    """
    基于快速行进法计算3D血管中心线（CPU版本）。
    参数:
        binary: 3D二值numpy数组，血管体积（1表示血管，0表示背景）
        dt: 3D浮点numpy数组，对应binary的距离变换值（距边界的距离）
        initial_seeds: 初始中心线种子点列表或数组（由坍缩前沿检测得到的点坐标）
        tau: 控制速度函数的参数τ（默认0.5，可根据血管半径范围调整）
    返回:
        centerline_points: 血管中心线的体素坐标列表（每个点为(z,y,x)元组）
    """
    # 确保数据类型
    binary = binary.astype(np.uint8)
    dt = dt.astype(np.float32)
    # 若未提供初始种子，则使用DT中的最大点作为起点
    if initial_seeds is None or len(initial_seeds) == 0:
        max_idx = np.unravel_index(np.argmax(dt * (binary > 0)), dt.shape)
        initial_seeds = [max_idx]
    else:
        initial_seeds = [tuple(pt) for pt in np.asarray(initial_seeds)]
    # 选取DT值最大的种子点作为起始点v [oai_citation_attribution:0‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=Algorithm%201%20Recover%20Centerline%20,v%2C%20DTB)
    seed_dt_vals = [dt[pt] for pt in initial_seeds]
    v_index = int(np.argmax(seed_dt_vals))
    v = initial_seeds[v_index]
    # 计算速度图像F(x)=exp(β * DT(x))，其中β=1/(max(DT)*τ) [oai_citation_attribution:1‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=where%20the%20driving%20force%20used,4)
    max_dt = float(np.max(dt))
    beta = 1.0 / (max_dt * tau) if max_dt > 0 else 0.0
    # 为保证数值精度，使用float64存储F和时间值，避免范围0-18内浮点误差
    F = np.zeros(dt.shape, dtype=np.float64)
    F[binary == 1] = np.exp(beta * dt[binary == 1].astype(np.float64))
    # 初始化到达时间数组，v点时间设为0，其余为正无限大
    times = np.full(dt.shape, np.inf, dtype=np.float64)
    times[v] = 0.0
    # 最小堆初始化（存储元素：(time, z, y, x)）
    pq = [(0.0, v[0], v[1], v[2])]
    # 邻域相对坐标6连通（不考虑对角，提高效率）
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    visited = np.zeros(dt.shape, dtype=bool)  # 标记已确定最短路径的点
    
    # **快速行进主循环**：求解Eikonal方程 ||∇T|| = 1/F(x) 的离散解 [oai_citation_attribution:2‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=agation%20process%20is%20computed%2C%20solving,the%20Eikonal%20equation)
    while pq:
        t, z, y, x = heapq.heappop(pq)
        # 跳过已更新的过期节点
        if t > times[z, y, x]:
            continue
        if visited[z, y, x]:
            continue
        visited[z, y, x] = True
        # 松弛当前点邻居的到达时间
        for dz, dy, dx in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx
            # 限制在图像范围内且在血管内部
            if nz < 0 or nz >= dt.shape[0] or ny < 0 or ny >= dt.shape[1] or nx < 0 or nx >= dt.shape[2]:
                continue
            if binary[nz, ny, nx] == 0 or visited[nz, ny, nx]:
                continue
            # 计算从当前点到邻居的行进时间增量：采用两点平均速度的倒数作为距离
            time_increment = 2.0 / (F[z, y, x] + F[nz, ny, nx] + 1e-12)
            new_t = times[z, y, x] + time_increment
            if new_t < times[nz, ny, nx]:
                times[nz, ny, nx] = new_t
                heapq.heappush(pq, (new_t, nz, ny, nx))
    
    # **中心线路径回溯**：从距离场中提取最小代价路径 [oai_citation_attribution:3‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=Then%2C%20a%20backtrace%20from%20m0,to%20v%20is%20performed%2C%20following) [oai_citation_attribution:4‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=v%20is%20obtained%2C%20and%20the,branches%20have%20to%20be%20com)
    centerline_points = []
    Sc = set(initial_seeds)  # 剩余初始种子集合（Sc）
    Sc.add(v)
    # 反复从剩余种子中找到 G 值最大的点，回溯到起点 v [oai_citation_attribution:5‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=Algorithm%201%20Recover%20Centerline%20,v%2C%20DTB)
    while Sc:
        # 选取当前种子集合Sc中到达时间G最大的点mi [oai_citation_attribution:6‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=5%3A%20mi%20%3D%20x%20,v%2Cmi%2CG%2C%20Sf%20%2C%20Sc)
        m = max(Sc, key=lambda pt: times[pt] if np.isfinite(times[pt]) else -np.inf)
        if not np.isfinite(times[m]):
            # 若某个种子点在G中不可达，则跳过
            Sc.remove(m)
            continue
        # 从mi回溯到v，沿着G的最小成本路径 [oai_citation_attribution:7‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=Then%2C%20a%20backtrace%20from%20m0,to%20v%20is%20performed%2C%20following)
        current = m
        branch_path = []
        while True:
            branch_path.append(current)
            # 若current在Sc中，标记已访问并移除，避免重复处理
            if current in Sc:
                Sc.remove(current)
            # 到达起点v或遇到已有中心线，则结束该分支回溯 [oai_citation_attribution:8‡file-munzesvapfkrc9csv1ryfq](file://file-MUnzesvAPfKrC9CSV1RyFQ#:~:text=v%20is%20obtained%2C%20and%20the,branches%20have%20to%20be%20com)
            if current == v or current in centerline_points:
                break
            # 从current的邻居中找下一步：选择具有更小到达时间的邻居（离v更近）
            cz, cy, cx = current
            next_pt = None
            min_t = times[current]
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if nz < 0 or nz >= dt.shape[0] or ny < 0 or ny >= dt.shape[1] or nx < 0 or nx >= dt.shape[2]:
                    continue
                if binary[nz, ny, nx] == 0 or not visited[nz, ny, nx]:
                    continue
                if times[nz, ny, nx] < min_t:  # 邻居更靠近v（G值更小）
                    min_t = times[nz, ny, nx]
                    next_pt = (nz, ny, nx)
            if next_pt is None:
                break  # 理论上应能找到，除非遇到无法回溯的情况
            current = next_pt
        # 将该分支路径加入最终中心线集合
        centerline_points.extend(branch_path)
    # 去除重复点，保持中心线点的一致性
    unique_centerline = []
    seen = set()
    for pt in centerline_points:
        if pt not in seen:
            unique_centerline.append(pt)
            seen.add(pt)
    return unique_centerline

# **用法示例**（与原process_vessel集成）
# centerline = fast_marching_centerline_cpu(stack, dt, seeds, tau=0.5)
# 其中stack为血管二值图像，dt为其距离变换，seeds为检测到的坍缩前沿点坐标列表


# # 使用ITK的快速行进算法（使用别名替换模板方式）
# def itk_fast_marching(seeds, dt_image):
#     Dimension = 3
#     # 使用 ITK 别名表示 3D 浮点图像
#     ImageType = itk.ImageF3  
#     # 注意维度顺序转换 (z,y,x) -> (x,y,z)
#     dt_transposed = dt_image.astype(np.float32).transpose(2, 1, 0)
#     itk_dt = itk.image_from_array(dt_transposed)
#     itk_dt.SetSpacing([1.0, 1.0, 1.0])
    
#     # 生成速度图像（这里使用 SigmoidImageFilterF3 别名）
#     sigmoid_filter = itk.SigmoidImageFilterF3.New()
#     sigmoid_filter.SetInput(itk_dt)
#     sigmoid_filter.SetAlpha(0.5)
#     sigmoid_filter.SetBeta(1.0)
#     sigmoid_filter.Update()
#     speed_image = sigmoid_filter.GetOutput()
    
#     # 设置种子点，使用 ITK 中 LevelSetNodeF3（代表 3D 浮点节点）
#     seeds_container = itk.VectorContainer[itk.UI, itk.LevelSetNodeF3].New()
#     seeds_container.Initialize()
#     for i, seed in enumerate(seeds):
#         node = itk.LevelSetNodeF3()
#         idx = itk.Index[3]()
#         # 坐标转换 (z,y,x) -> (x,y,z)
#         idx[0] = int(seed[2])
#         idx[1] = int(seed[1])
#         idx[2] = int(seed[0])
#         node.SetIndex(idx)
#         node.SetValue(0.0)
#         seeds_container.InsertElement(i, node)
#     print(f"[DEBUG] ITK fast marching: Set {len(seeds)} seed(s).")
    
#     # 使用 FastMarchingImageFilterF3 别名
#     fm_filter = itk.FastMarchingImageFilterF3.New()
#     fm_filter.SetInput(speed_image)
#     fm_filter.SetTrialPoints(seeds_container)
#     fm_filter.SetOutputSize(itk_dt.GetLargestPossibleRegion().GetSize())
#     fm_filter.SetOutputSpacing(itk_dt.GetSpacing())
#     fm_filter.SetOutputOrigin(itk_dt.GetOrigin())
#     stopping_value = float(np.max(dt_image) * 3.0)
#     fm_filter.SetStoppingValue(stopping_value)
#     fm_filter.Update()
    
#     arrival_time = fm_filter.GetOutput()
#     # 将 ITK 图像转换回 numpy 数组，并恢复维度顺序 (x,y,z) -> (z,y,x)
#     arrival_array = itk.array_from_image(arrival_time).transpose(2, 1, 0)
#     max_val = np.finfo(np.float32).max
#     arrival_array[arrival_array > max_val / 2] = np.inf
#     print("[DEBUG] ITK fast marching completed. Arrival time map shape: {}".format(arrival_array.shape))
#     return arrival_array

# # ITK 路径反向追踪
# def itk_backtrack(start, end, arrival_time):
#     Dimension = 3
#     ImageType = itk.ImageF3
#     arr_transposed = arrival_time.astype(np.float32).transpose(2, 1, 0)
#     itk_arrival = itk.image_from_array(arr_transposed)
#     itk_arrival.SetSpacing([1.0, 1.0, 1.0])
    
#     # 转换起点与终点坐标 (z,y,x) -> (x,y,z)
#     start_idx = itk.Index[3]()
#     start_idx[0] = int(start[2])
#     start_idx[1] = int(start[1])
#     start_idx[2] = int(start[0])
    
#     end_idx = itk.Index[3]()
#     end_idx[0] = int(end[2])
#     end_idx[1] = int(end[1])
#     end_idx[2] = int(end[0])
    
#     # 创建路径查找滤波器
#     path_filter = itk.SpeedFunctionToPathFilter[ImageType, itk.PolyLineParametricPath[3]].New()
#     path_filter.SetInput(itk_arrival)
#     path_filter.SetPathStartPoint(start_idx)
#     path_filter.SetPathEndPoint(end_idx)
#     path_filter.SetUseImageSpacing(True)
#     path_filter.Update()
    
#     path = path_filter.GetOutput(0)
#     path_points = []
#     n_points = 100  # 调整路径点密度
#     for i in range(n_points):
#         point = path.Evaluate(float(i) / float(n_points-1))
#         # 转换回 (z,y,x) 坐标系
#         path_points.append((int(point[2]), int(point[1]), int(point[0])))
#     print("[DEBUG] Backtracked path using ITK. Total path points: {}".format(len(path_points)))
#     return path_points

# 主流程
def process_vessel(input_filepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] Loading stack...")
    stack = load_tiff_stack(input_filepath)
    
    print("[INFO] Computing boundary...")
    boundary = compute_3d_boundary(stack)
    
    print("[INFO] Computing distance transform...")
    dt = compute_3d_distance_transform(stack)
    
    print("[INFO] Detecting collapsing fronts...")
    collapsing_points = detect_3d_collapsing_fronts(dt)

    print("[INFO] Running fast marching algorithm...")
    seeds = np.argwhere(collapsing_points)
    centerline = fast_marching_centerline_cpu(stack, dt, seeds, tau=0.5)
    
    # # 使用 collapsing_points 作为种子；若未检测到则退回到 dt 最大点
    # seeds = np.argwhere(collapsing_points)
    # if seeds.size == 0:
    #     print("[WARNING] No collapsing points detected, using dt maximum as seed.")
    #     seeds = np.argwhere(dt == np.max(dt))
    # else:
    #     print(f"[DEBUG] Using {seeds.shape[0]} collapsing point(s) as seeds.")
    
    # print("[INFO] Running ITK fast marching...")
    # arrival_time = itk_fast_marching(seeds, dt)
    
    # print("[INFO] Backtracking path...")
    # valid_mask = np.isfinite(arrival_time)
    # if not np.any(valid_mask):
    #     raise ValueError("No valid path found in arrival time map.")
    
    # end_point = np.unravel_index(np.nanargmax(arrival_time), arrival_time.shape)
    # start_point = tuple(seeds[0])
    # print(f"[DEBUG] Start point: {start_point}, End point: {end_point}")
    
    # try:
    #     centerline = itk_backtrack(start_point, end_point, arrival_time)
    # except RuntimeError as e:
    #     print(f"[ERROR] Path finding failed: {str(e)}")
    #     return
    
    # 可视化与保存中间结果
    mid_slice = stack.shape[0] // 2
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(stack[mid_slice], cmap='gray')
    plt.title('Binary Slice')
    plt.subplot(122)
    plt.imshow(dt[mid_slice], cmap='jet')
    plt.title('Distance Transform')
    plt.colorbar()
    intermediate_filepath = os.path.join(output_dir, 'intermediate.png')
    plt.savefig(intermediate_filepath)
    print(f"[DEBUG] Intermediate results saved to {intermediate_filepath}")
    
    centerline_filepath = os.path.join(output_dir, 'centerline.npy')
    np.save(centerline_filepath, np.array(centerline))
    print(f"[DEBUG] Centerline saved to {centerline_filepath}")
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    input_file = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif"
    output_folder = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/dsresults"
    process_vessel(input_file, output_folder)