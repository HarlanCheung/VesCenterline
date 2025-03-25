import numpy as np
from skimage.morphology import skeletonize_3d
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt

import os
import tifffile

def load_tiff_stack(path):
    return tifffile.imread(path)

def save_tiff_stack(array, path):
    array_uint8 = (array.astype(np.uint8)) * 255  # 转为0和255
    tifffile.imwrite(path, array_uint8)

def extract_skeleton(volume):
    skeleton = skeletonize_3d(volume)
    return skeleton

def remove_small_branches(skeleton, length_threshold=10):
    # 连通域标记
    structure = generate_binary_structure(3, 3)
    labeled, num = label(skeleton, structure=structure)

    # 初始化修剪后的骨架
    pruned = np.zeros_like(skeleton, dtype=bool)

    for i in range(1, num + 1):
        component = (labeled == i)
        if np.sum(component) >= length_threshold:
            pruned |= component  # 保留该分支

    return pruned

def main():
    # === 修改为你的路径 ===
    input_path = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif"  # 输入二值化图像栈路径
    skel_output_path = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/skraw.tif"
    pruned_output_path = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/postsk.tif"

    # === 读取数据 ===
    binary_volume = load_tiff_stack(input_path)
    print("Loaded binary volume with shape:", binary_volume.shape)

    # === 阶段一：骨架提取 ===
    skeleton = extract_skeleton(binary_volume)
    save_tiff_stack(skeleton, skel_output_path)
    print("Saved raw skeleton to:", skel_output_path)

    # === 阶段二：剪枝处理 ===
    pruned_skeleton = remove_small_branches(skeleton, length_threshold=10)
    save_tiff_stack(pruned_skeleton, pruned_output_path)
    print("Saved pruned skeleton to:", pruned_output_path)

if __name__ == "__main__":
    main()