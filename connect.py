import SimpleITK as sitk
import matplotlib.pyplot as plt

def connected_component_bar_plot(image_path):
    # 读取 TIFF 格式二值图像（假设图像为 0/1 二值化数据）
    itk_image = sitk.ReadImage(image_path)
    
    # 连通体标记，采用完全连通性
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    cc_image = cc_filter.Execute(itk_image)
    
    # 利用 LabelShapeStatisticsImageFilter 获取每个连通体的体素数
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(cc_image)
    
    voxel_counts = []
    labels = label_stats.GetLabels()  # 返回所有连通体标签
    # 遍历每个连通体，获取其体素数
    for label in labels:
        count = label_stats.GetNumberOfPixels(label)
        voxel_counts.append(count)
    
    # 绘制条形图：x 轴为连通体编号（离散），y 轴为对应体素数
    plt.figure(figsize=(10, 6))
    x = list(range(1, len(voxel_counts) + 1))
    plt.bar(x, voxel_counts, align='center', edgecolor='black')
    plt.xlabel('连通体编号')
    plt.ylabel('体素数')
    plt.title('连通体体素数分布')
    plt.xticks(x)  # 使 x 轴刻度为离散的连通体编号
    plt.show()

if __name__ == '__main__':
    # 修改为你的 TIFF 文件路径
    image_path = '/Users/harlan/Documents/shaolab/code_proj/centerline/VesCenterline/ssp_nkeptstack.tif'
    connected_component_bar_plot(image_path)