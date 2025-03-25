import itk

import sys

# 打印ITK版本
print(f"ITK版本: {itk.Version.GetITKVersion()}")

# 列出已安装的ITK模块
print("\n已安装的ITK模块:")
for module in dir(itk):
    if not module.startswith('_'):
        print(f"- {module}")

# 查找与路径相关的类和函数
print("\n路径提取相关的类:")
path_related = [item for item in dir(itk) if 'Path' in item]
for item in path_related:
    print(f"- {item}")

# 检查是否可以创建最小路径相关的过滤器
try:
    ImageType = itk.Image[itk.F, 3]
    PathType = itk.PolyLineParametricPath[3]
    
    # 尝试创建路径过滤器
    path_filter = itk.SpeedFunctionToPathFilter[ImageType, PathType].New()
    print("\n成功创建SpeedFunctionToPathFilter实例!")
    
    # 检查其他相关类
    if hasattr(itk, 'ArrivalFunctionToPathFilter'):
        print("成功找到ArrivalFunctionToPathFilter类!")
    
except Exception as e:
    print(f"\n创建过滤器时出错: {str(e)}")

print("\n模块检查完成。")