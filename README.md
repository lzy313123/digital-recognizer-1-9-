# 手写数字识别系统

基于 PyTorch 和 Tkinter 的简单手写数字识别应用。  
使用 MNIST 数据集训练一个卷积神经网络（CNN），并提供画板界面，让用户用鼠标手写数字，实时识别。

---

## 技术栈

- **深度学习框架**: PyTorch 1.x  
- **计算机视觉**: torchvision, OpenCV  
- **GUI**: Tkinter (Python 标准库)  
- **图像处理**: Pillow, NumPy  
- **数据增强**: torchvision.transforms.RandomAffine  

---

## 文件结构
├── 识别数字.py # 训练脚本（CNN网络，数据增强）
├── digital_draw.py # 画板识别程序（加载模型，实时预测）
├── mnist_cnn.pth # 训练好的模型权重（需自行训练生成）
├── .gitignore # Git忽略文件（Python模板）
├── LICENSE # MIT许可证
└── README.md # 项目说明文档

---

## 核心功能

### 1. 模型训练 (`识别数字.py`)
- 使用 **卷积神经网络（CNN）** 架构：
  - 2个卷积层 + 最大池化
  - 全连接层 + Dropout 正则化
- 训练数据增强：随机旋转、平移、缩放，提高泛化能力。
- 在 MNIST 测试集上达到约 **98%** 的准确率。
- 训练完成后自动保存模型为 `mnist_cnn.pth`。

### 2. 手写识别画板 (`digital_draw.py`)
- 提供 280×280 的白底画布，黑色画笔（宽度可调）。
- 点击“识别”按钮，程序会：
  1. 将画板图像转为灰度并二值化。
  2. 自动裁剪数字区域，并居中缩放到 28×28。
  3. 进行与训练相同的归一化（均值 0.1307，标准差 0.3081）。
  4. 送入模型预测，显示结果。
- 支持“清除”和“退出”按钮。

### 3. 预处理优化
- **自适应二值化**（Otsu 阈值）适应不同亮度。
- **数字居中**：自动计算数字边界并移到中央，消除位置偏移影响。
- **尺寸归一化**：小数字自动放大到 20×20 左右，保证结构清晰。

---

## 使用说明

### 环境配置
```bash
# 安装依赖
pip install torch torchvision matplotlib opencv-python pillow numpy

#训练模型
bash
python 识别数字.py
首次运行会自动下载 MNIST 数据集（约 60 MB），训练约 10 个 epoch 后生成 mnist_cnn.pth。

#启动画板
bash
python digital_draw.py
弹出窗口后，用鼠标绘制数字（按住左键拖动），点击“识别”按钮即可看到结果。

###参数调节
画笔粗细：在 digital_draw.py 的 paint 方法中修改 width 值（默认 8）。

小数字增强：修改 preprocess_image 中的 min_size（默认 20），数值越大数字主体越大。

二值化阈值：可将 Otsu 自适应阈值改为固定值，如 cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY_INV)。

###常见问题
#画板识别不准怎么办？

确保绘制的数字清晰，尽量写大一些。

可调整 min_size 或二值化阈值。

#若经常出现误判，可增加训练时的数据增强范围（修改 scale 和 translate）。

#模型加载失败？

确认 mnist_cnn.pth 与脚本在同一目录。

如果重新训练过，请确保网络结构与保存的模型一致。


###许可证
本项目采用 MIT 许可证，详情见 LICENSE 文件。


