import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os

def normalize_image(image, dtype=np.uint8):
    """Normalize image data to 0-255 for uint8 or 0-65535 for uint16."""
    image = image - np.min(image)
    image = image / np.max(image)
    if dtype == np.uint8:
        image = (image * 255).astype(np.uint8)
    elif dtype == np.uint16:
        image = (image * 65535).astype(np.uint16)
    return image

def npy_to_tiff(npy_path, output_path, dtype=np.uint8, show=False):
    """Convert an .npy file to .tiff and optionally visualize it."""
    image = np.load(npy_path)
    print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
    image = normalize_image(image, dtype=dtype)
    
    tiff.imwrite(output_path, image)
    print(f"Saved TIFF file to: {output_path}")
    
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.title("Converted TIFF Preview")
        plt.show()

if __name__ == "__main__":
    # 修改为你本地的 .npy 文件路径
    npy_file = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/dsresults/centerline.npy"  # 替换为你的 .npy 文件路径
    output_tiff = "/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/dsresults/centerline.tiff"  # 替换为你的 .tiff 文件路径
    
    # 选择数据类型（uint8 或 uint16）
    dtype = np.uint8  # 或者 np.uint16
    
    # 是否显示转换后的图像
    show_image = True
    
    npy_to_tiff(npy_file, output_tiff, dtype=dtype, show=show_image)
