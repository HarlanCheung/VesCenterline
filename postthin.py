import numpy as np
import tifffile
from skimage.morphology import skeletonize_3d
from scipy.ndimage import label

def load_tiff_stack(filename):
    """
    读取 tiff 堆栈数据，返回 numpy 数组（二值化后）。
    """
    stack = tifffile.imread(filename)
    # 简单阈值化（根据实际数据调整阈值）
    binary_stack = (stack > 128).astype(np.uint8)
    return binary_stack

def count_neighbors(skel, z, y, x):
    """
    计算 skel 中 (z,y,x) 处在 26 邻域内的骨架体素数量。
    """
    zmin, zmax = max(z-1, 0), min(z+2, skel.shape[0])
    ymin, ymax = max(y-1, 0), min(y+2, skel.shape[1])
    xmin, xmax = max(x-1, 0), min(x+2, skel.shape[2])
    return np.sum(skel[zmin:zmax, ymin:ymax, xmin:xmax]) - 1

def is_simple_point(neighborhood):
    """
    判断删除中心点后，局部 (3x3x3) 区域的骨架是否仍然连通。
    如果删除中心点后，剩余骨架体素只有一个连通块，则认为该点是简单的。
    """
    structure = np.ones((3,3,3), dtype=np.uint8)
    labeled, num = label(neighborhood, structure=structure)
    return num == 1

def attempt_remove_voxel(skel, z, y, x):
    """
    尝试删除 skel 中 (z,y,x) 处的体素。
    检查局部 3x3x3 区域（必要时自动考虑边界），
    若删除后局部连通性不受破坏，则返回 True，表明可以删除。
    """
    zmin = z-1 if z-1 >= 0 else 0
    zmax = z+2 if z+2 <= skel.shape[0] else skel.shape[0]
    ymin = y-1 if y-1 >= 0 else 0
    ymax = y+2 if y+2 <= skel.shape[1] else skel.shape[1]
    xmin = x-1 if x-1 >= 0 else 0
    xmax = x+2 if x+2 <= skel.shape[2] else skel.shape[2]
    
    local = skel[zmin:zmax, ymin:ymax, xmin:xmax].copy()
    cz, cy, cx = z - zmin, y - ymin, x - xmin
    local[cz, cy, cx] = 0  # 模拟删除中心点

    if np.sum(local) == 0:
        return True
    return is_simple_point(local)

def post_process_skeleton(skel, max_iter=50):
    """
    反复迭代骨架后处理：
    每次遍历骨架体素，若某体素的 26 邻域内骨架体素数量大于 3，
    则尝试删除该体素（仅在局部连通性保持的前提下）。
    迭代直到没有体素违反要求或达到最大迭代次数。
    """
    iter_count = 0
    while iter_count < max_iter:
        to_delete = []
        coords = np.argwhere(skel == 1)
        # 遍历所有骨架体素
        for z, y, x in coords:
            ncount = count_neighbors(skel, z, y, x)
            if ncount > 3:
                if attempt_remove_voxel(skel, z, y, x):
                    to_delete.append((z, y, x))
        if not to_delete:
            print(f"迭代结束，共迭代 {iter_count} 次，无需删除体素。")
            break
        # 删除满足条件的体素
        for z, y, x in to_delete:
            skel[z, y, x] = 0
        iter_count += 1
        print(f"迭代 {iter_count}: 删除体素 {len(to_delete)} 个")
    return skel

def analyze_skeleton(skel):
    """
    遍历骨架体素，统计各类邻域（1 邻、2 邻、3 邻、>3 邻）体素数量。
    """
    coords = np.argwhere(skel == 1)
    neighbor_counts = [count_neighbors(skel, z, y, x) for z, y, x in coords]
    neighbor_counts = np.array(neighbor_counts)
    stats = {
        'total': len(neighbor_counts),
        '1_neighbor': int(np.sum(neighbor_counts == 1)),
        '2_neighbor': int(np.sum(neighbor_counts == 2)),
        '3_neighbor': int(np.sum(neighbor_counts == 3)),
        '>3_neighbor': int(np.sum(neighbor_counts > 3))
    }
    return stats

if __name__ == '__main__':
    # 1. 载入 tiff 堆栈数据（请将 'test_stack.tif' 替换为实际文件路径）
    filename = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif'
    volume = load_tiff_stack(filename)
    print("原始体数据 shape:", volume.shape)

    # 2. 初步骨架提取
    skel = skeletonize_3d(volume > 0)
    skel = skel.astype(np.uint8)
    stats = analyze_skeleton(skel)
    print("初步骨架统计:", stats)

    # 3. 迭代后处理：多次迭代，直到骨架中不再存在超过 3 邻的体素
    skel_processed = post_process_skeleton(skel, max_iter=50)
    stats_after = analyze_skeleton(skel_processed)
    print("后处理后骨架统计:", stats_after)

    # 可选：保存处理后的骨架为 tiff 堆栈，便于后续观察
    tifffile.imwrite('/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/postcenterline.tif', skel_processed.astype(np.uint8))