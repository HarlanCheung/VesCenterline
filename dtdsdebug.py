import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, convolve
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import itk

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

# 使用 ITK 的快速行进算法
def itk_fast_marching(seeds, dt_image):
    Dimension = 3
    PixelType = itk.F
    ImageType = itk.Image[PixelType, Dimension]
    
    # 转置维度 (z,y,x) -> (x,y,z)
    dt_transposed = dt_image.astype(np.float32).transpose(2, 1, 0)
    itk_dt = itk.image_from_array(dt_transposed)
    itk_dt.SetSpacing([1.0, 1.0, 1.0])
    
    # 使用 SigmoidImageFilter 创建速度图像
    sigmoid_filter = itk.SigmoidImageFilter[ImageType, ImageType].New()
    sigmoid_filter.SetInput(itk_dt)
    sigmoid_filter.SetAlpha(0.5)
    sigmoid_filter.SetBeta(1.0)
    sigmoid_filter.Update()
    speed_image = sigmoid_filter.GetOutput()
    
    # 设置种子点，采用 LevelSetNode 模板方式
    seeds_container = itk.VectorContainer[itk.UI, itk.LevelSetNode[PixelType, Dimension]].New()
    seeds_container.Initialize()
    for i, seed in enumerate(seeds):
        node = itk.LevelSetNode[PixelType, Dimension]()
        idx = itk.Index[Dimension]()
        # 坐标转换 (z,y,x) -> (x,y,z)
        idx[0] = int(seed[2])
        idx[1] = int(seed[1])
        idx[2] = int(seed[0])
        node.SetIndex(idx)
        node.SetValue(0.0)
        seeds_container.InsertElement(i, node)
    print(f"[DEBUG] ITK fast marching: Set {len(seeds)} seed(s).")
    
    # 配置 FastMarchingImageFilter
    fm_filter = itk.FastMarchingImageFilter[ImageType, ImageType].New()
    fm_filter.SetInput(speed_image)
    fm_filter.SetTrialPoints(seeds_container)
    fm_filter.SetOutputSize(itk_dt.GetLargestPossibleRegion().GetSize())
    fm_filter.SetOutputSpacing(itk_dt.GetSpacing())
    fm_filter.SetOutputOrigin(itk_dt.GetOrigin())
    stopping_value = float(np.max(dt_image) * 3.0)
    fm_filter.SetStoppingValue(stopping_value)
    fm_filter.Update()
    
    arrival_time = fm_filter.GetOutput()
    # 转换回 numpy 数组，并恢复维度顺序 (x,y,z) -> (z,y,x)
    arrival_array = itk.array_from_image(arrival_time).transpose(2, 1, 0)
    max_val = np.finfo(np.float32).max
    arrival_array[arrival_array > max_val / 2] = np.inf
    print("[DEBUG] ITK fast marching completed. Arrival time map shape: {}".format(arrival_array.shape))
    return arrival_array

# ITK 路径反向追踪
def itk_backtrack(start, end, arrival_time):
    # 将 arrival_time 从 numpy 数组转换为 ITK 图像
    Dimension = 3
    ImageType = itk.Image[itk.F, Dimension]
    
    # 转换维度顺序 (z,y,x) -> (x,y,z)
    arr_transposed = arrival_time.astype(np.float32).transpose(2, 1, 0)
    itk_arrival = itk.image_from_array(arr_transposed)
    itk_arrival.SetSpacing([1.0, 1.0, 1.0])
    
    # 转换为ITK坐标格式
    start_idx = itk.Index[Dimension]()
    start_idx[0] = int(start[2])  # x
    start_idx[1] = int(start[1])  # y
    start_idx[2] = int(start[0])  # z
    
    end_idx = itk.Index[Dimension]()
    end_idx[0] = int(end[2])
    end_idx[1] = int(end[1])
    end_idx[2] = int(end[0])
    
    # 创建终点信息
    end_point_info = itk.SpeedFunctionPathInformation[itk.F, Dimension]()
    end_point_info.SetEndPoint(end_idx)
    
    # 创建路径查找滤波器
    path_filter = itk.SpeedFunctionToPathFilter[ImageType, itk.PolyLineParametricPath[Dimension]].New()
    path_filter.SetInput(itk_arrival)
    path_filter.SetTerminationValue(0.0)
    
    # 添加路径信息
    path_filter.ClearPathInfo()
    path_filter.AddPathInformation(end_point_info)
    
    # 使用SetStartPoint代替SetPathStartPoint
    path_filter.SetStartPoint(start_idx)
    
    # 更新过滤器
    path_filter.Update()
    
    # 获取路径点
    if path_filter.GetNumberOfOutputs() > 0:
        path = path_filter.GetOutput(0)
        
        # 转换为点列表
        path_points = []
        n_points = 100  # 可以调整点的密度
        for i in range(n_points):
            t = float(i) / (n_points - 1)
            point = path.Evaluate(t)
            # 转换回 (z,y,x) 坐标系
            path_points.append((int(point[2]), int(point[1]), int(point[0])))
        
        return path_points
    else:
        raise RuntimeError("路径提取失败，没有可用的输出路径")

# 主函数：先加载（或生成） dt 与种子数据，然后仅调用 ITK 部分函数
if __name__ == "__main__":
    # 路径设置
    input_file = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif"
    output_folder = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/dsresults"
    os.makedirs(output_folder, exist_ok=True)
    
    # 定义保存 dt 和 seeds 的文件路径
    dt_file = os.path.join(output_folder, "dt.npy")
    seeds_file = os.path.join(output_folder, "seeds.npy")
    
    # 如果之前保存过 dt 和 seeds，则直接加载，否则计算并保存
    if os.path.exists(dt_file) and os.path.exists(seeds_file):
        print("[INFO] Loading saved dt and seeds data.")
        dt = np.load(dt_file)
        seeds = np.load(seeds_file)
    else:
        print("[INFO] Computing dt and seeds data from the input stack...")
        stack = load_tiff_stack(input_file)
        dt = compute_3d_distance_transform(stack)
        collapsing_points = detect_3d_collapsing_fronts(dt)
        seeds = np.argwhere(collapsing_points)
        if seeds.size == 0:
            print("[WARNING] No collapsing points detected, using dt maximum as seed.")
            seeds = np.argwhere(dt == np.max(dt))
        else:
            print(f"[DEBUG] Using {seeds.shape[0]} collapsing point(s) as seeds.")
        np.save(dt_file, dt)
        np.save(seeds_file, seeds)
        print(f"[DEBUG] Saved dt to {dt_file} and seeds to {seeds_file}")
    
    # 仅调用后面的 ITK 函数进行调试
    print("[INFO] Running ITK fast marching...")
    arrival_time = itk_fast_marching(seeds, dt)
    
    print("[INFO] Backtracking path...")
    start_point = tuple(seeds[0])
    end_point = np.unravel_index(np.nanargmax(arrival_time), arrival_time.shape)
    print(f"[DEBUG] Start point: {start_point}, End point: {end_point}")
    centerline = itk_backtrack(start_point, end_point, arrival_time)
    print("[INFO] Centerline extracted.")
    print("Centerline:", centerline)