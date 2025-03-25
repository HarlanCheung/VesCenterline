import numpy as np
import scipy.ndimage as ndi
import tifffile

def match(cube):
  """Match one of the masks in the algorithm.
  
  Arguments
  ---------
  cube : 3x3x3 bool array
    The local binary image.
  
  Returns
  -------
  match : bool
    True if one of the masks matches
  
  Note
  ----
  Algorithm as in Palagyi & Kuba (1999)
  """
  #T1
  T1 = (cube[1,1,0] & cube[1,1,1] & 
        (cube[0,0,0] or cube[1,0,0] or cube[2,0,0] or
         cube[0,1,0] or cube[2,1,0] or
         cube[0,2,0] or cube[1,2,0] or cube[2,2,0] or
         cube[0,0,1] or cube[1,0,1] or cube[2,0,1] or
         cube[0,1,1] or cube[2,1,1] or 
         cube[0,2,1] or cube[1,2,1] or cube[2,2,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]))
  if T1:
    return True
  
  #T2
  T2 = (cube[1,1,1] & cube[1,2,1] & 
        (cube[0,1,0] or cube[1,1,0] or cube[2,1,0] or
         cube[0,2,0] or cube[1,2,0] or cube[2,2,0] or
         cube[0,1,1] or cube[2,1,1] or
         cube[0,2,1] or cube[2,2,1] or
         cube[0,1,2] or cube[1,1,2] or cube[2,1,2] or
         cube[0,2,2] or cube[1,2,2] or cube[2,2,2]) &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) &
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]))
  if T2: 
    return True
    
  #T3
  T3 = (cube[1,1,1] & cube[1,2,0] & 
        (cube[0,1,0] or cube[2,1,0] or
         cube[0,2,0] or cube[2,2,0] or
         cube[0,1,1] or cube[2,1,1] or
         cube[0,2,1] or cube[2,2,1]) &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) &
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & ( not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]))
  if T3:
    return True
  
  #T4
  T4 = (cube[1,1,0] & cube[1,1,1] & cube[1,2,1] & 
        ((not cube[0,0,1]) or (not cube[0,1,2])) &
        ((not cube[2,0,1]) or (not cube[2,1,2])) &
        (not cube[1,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]))
  if T4:
    return True  
  
  #T5
  T5 = (cube[1,1,0] & cube[1,1,1] & cube[1,2,1] & cube[2,0,2] &
        ((not cube[0,0,1]) or (not cube[0,1,2])) &
        (((not cube[2,0,1]) & cube[2,1,2]) or (cube[2,0,1] & (not cube[2,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) &
        (not cube[1,1,2]))
  if T5:
    return True
    
  #T6
  T6 = (cube[1,1,0] & cube[1,1,1] & cube[1,2,1] & cube[0,0,2] &
        ((not cube[2,0,1]) or (not cube[2,1,2])) &
        (((not cube[0,0,1]) & cube[0,1,2]) or (cube[0,0,1] & (not cube[0,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]))
  if T6:
    return True
    
  #T7
  T7 = (cube[1,1,0] & cube[1,1,1] & cube[2,1,1] &  cube[1,2,1] &
        ((not cube[0,0,1]) or (not cube[0,1,2])) &
        (not cube[1,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) &
        (not cube[1,1,2]))
  if T7:
    return True
  
  #T8
  T8 = (cube[1,1,0] & cube[0,1,1] & cube[1,1,1] & cube[1,2,1] &
        ((not cube[2,0,1]) or (not cube[2,1,2])) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]))
  if T8:
    return True 
    
  #T9
  T9 = (cube[1,1,0] & cube[1,1,1] & cube[2,1,1] & cube[0,0,2] & cube[1,2,1] &
        (((not cube[0,0,1]) & cube[0,1,2]) or (cube[0,0,1] & (not cube[0,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) &
        (not cube[1,1,2]))
  if T9:
    return True  
    
  #T10
  T10= (cube[1,1,0] & cube[0,1,1] & cube[1,1,1] & cube[2,0,2] & cube[1,2,1] &
        (((not cube[2,0,1]) & cube[2,1,2]) or (cube[2,0,1] & (not cube[2,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) &
        (not cube[1,1,2]))
  if T10:
    return True  
    
  #T11
  T11= (cube[2,1,0] & cube[1,1,1] & cube[1,2,0] &
        (not cube[0,0,0]) & (not cube[1,0,0]) & 
        (not cube[0,0,1]) & (not cube[1,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]))
  if T11: 
    return True
    
  #T12
  T12= (cube[0,1,0] & cube[1,2,0] & cube[1,1,1] &
        (not cube[1,0,0]) & (not cube[2,0,0]) & 
        (not cube[1,0,1]) & (not cube[2,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]))
  if T12: 
    return True
    
  #T13
  T13= (cube[1,2,0] & cube[1,1,1] & cube[2,2,1] &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]))
  if T13: 
    return True 
    
  #T14
  T14= (cube[1,2,0] & cube[1,1,1] & cube[0,2,1] &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[1,2,2]) & (not cube[2,2,2]))
  if T14: 
    return True 
    
  return False

def PK12_skeletonize(binary_image, max_iterations=500):
    """
    使用PK12算法对3D二值图像进行骨架提取
    
    参数:
    binary_image: 3D二值图像数组
    max_iterations: 最大迭代次数
    
    返回:
    skeleton: 提取后的骨架图像
    """
    skeleton = binary_image.copy()
    changed = True
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # 创建一个标记数组来存储要移除的点
        to_remove = np.zeros_like(skeleton, dtype=bool)
        
        # 填充边界以便处理边缘像素
        padded = np.pad(skeleton, 1, mode='constant', constant_values=0)
        
        # 只获取前景点坐标 - 这可以大大减少需要检查的点数
        foreground_coords = np.where(padded == 1)
        
        for i in range(len(foreground_coords[0])):
            z, y, x = foreground_coords[0][i], foreground_coords[1][i], foreground_coords[2][i]
            
            # 跳过边界上的点
            if (z == 0 or z == padded.shape[0]-1 or
                y == 0 or y == padded.shape[1]-1 or
                x == 0 or x == padded.shape[2]-1):
                continue
            
            # 提取3x3x3邻域，检查是否匹配任何模板
            cube = padded[z-1:z+2, y-1:y+2, x-1:x+2].copy()
            if match(cube):
                to_remove[z-1, y-1, x-1] = True
                changed = True
        
        # 移除匹配的点
        if np.any(to_remove):
            skeleton[to_remove] = 0
        else:
            break
            
        if iteration % 5 == 0:
            print(f"迭代 {iteration}: 移除了 {np.sum(to_remove)} 个点")
    
    print(f"骨架化完成，共执行了 {iteration} 次迭代")
    return skeleton

# 数据导入及预处理
def load_and_binarize_image(file_path, threshold=128):
    print(f"加载图像: {file_path}")
    image = tifffile.imread(file_path)
    print(f"图像尺寸: {image.shape}")
    binary_image = image > threshold
    return binary_image.astype(np.uint8)

# 骨架数据保存
def save_skeleton(skeleton, output_path):
    tifffile.imwrite(output_path, skeleton.astype(np.uint8)*255)
    print(f"骨架已保存至: {output_path}")
    print(f"骨架中非零点数量: {np.sum(skeleton)}")

# 主函数调用示例
if __name__ == "__main__":
    input_path = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif'
    output_path = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/PK12ske.tif'
    
    print("开始处理...")
    binary_image = load_and_binarize_image(input_path)
    print("开始骨架化...")
    skeleton = PK12_skeletonize(binary_image)
    save_skeleton(skeleton, output_path)
    
    print("处理完成")