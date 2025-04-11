from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取两张图片
image1 = Image.open('cpu_grad.png')
image2 = Image.open('gpu_grad.png')

# 确保两张图片的尺寸相同
if image1.size != image2.size:
    raise ValueError("两张图片的尺寸必须相同")

# 将图片转换为numpy数组
image1_array = np.array(image1)
image2_array = np.array(image2)

# 计算两张图片的差值的绝对值
abs_diff = np.abs(image1_array - image2_array)

# 显示结果
plt.imshow(abs_diff)
plt.title('Absolute Difference')
plt.axis('off')  # 关闭坐标轴
plt.show()