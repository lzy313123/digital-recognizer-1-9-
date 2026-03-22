在你的项目根目录下创建或修改 `README.md` 文件，内容如下：

```markdown
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

```
.
├── 识别数字.py              # 训练脚本（CNN网络，数据增强）
├── digital_draw.py          # 画板识别程序（加载模型，实时预测）
├── mnist_cnn.pth            # 训练好的模型权重（需自行训练生成）
├── .gitignore               # Git忽略文件（Python模板）
├── LICENSE                  # MIT许可证
└── README.md                # 项目说明文档
```
以下是优化后的“使用说明”部分，采用清晰的分点形式，代码块格式正确，并保留加粗标题，更接近 Word 风格：

```markdown
## 使用说明

### 1. 环境配置

安装所需依赖库：

```bash
pip install torch torchvision matplotlib opencv-python pillow numpy
```

### 2. 训练模型

运行训练脚本，自动下载 MNIST 数据集（约 60 MB）并开始训练：

```bash
python 识别数字.py
```

训练约 10 个 epoch 后，模型准确率可达到约 98%，并自动保存为 `mnist_cnn.pth` 文件。

### 3. 启动画板

运行画板程序，弹出绘制窗口：

```bash
python digital_draw.py
```

- 在白色画布上用鼠标绘制数字（按住左键拖动）。  
- 点击 **“识别”** 按钮，程序会进行预处理并显示识别结果。  
- 点击 **“清除”** 可以重新绘制，**“退出”** 关闭窗口。
```

将这段内容替换原 README 中对应的“使用说明”部分即可。

## 使用说明

### 环境配置
```bash
# 安装依赖
pip install torch torchvision matplotlib opencv-python pillow numpy
```

### 训练模型
```bash
python 识别数字.py
```
首次运行会自动下载 MNIST 数据集（约 60 MB），训练约 10 个 epoch 后生成 `mnist_cnn.pth`。

### 启动画板
```bash
python digital_draw.py
```
弹出窗口后，用鼠标绘制数字（按住左键拖动），点击“识别”按钮即可看到结果。

---

## 效果示例

| 绘制数字 | 识别结果 |
|----------|----------|
| ![手写3](https://via.placeholder.com/80?text=3) | 3 |
| ![手写8](https://via.placeholder.com/80?text=8) | 8 |
| ![手写7](https://via.placeholder.com/80?text=7) | 7 |

（实际效果请自行运行体验）

---

## 参数调节

- **画笔粗细**：在 `digital_draw.py` 的 `paint` 方法中修改 `width` 值（默认 8）。
- **小数字增强**：修改 `preprocess_image` 中的 `min_size`（默认 20），数值越大数字主体越大。
- **二值化阈值**：可将 Otsu 自适应阈值改为固定值，如 `cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY_INV)`。

---

## 常见问题

1. **画板识别不准怎么办？**  
   - 确保绘制的数字清晰，尽量写大一些。  
   - 可调整 `min_size` 或二值化阈值。  
   - 若经常出现误判，可增加训练时的数据增强范围（修改 `scale` 和 `translate`）。

2. **模型加载失败？**  
   - 确认 `mnist_cnn.pth` 与脚本在同一目录。  
   - 如果重新训练过，请确保网络结构与保存的模型一致。

3. **推送 GitHub 时模型文件过大？**  
   - 模型文件约 2-3 MB，可以直接推送。  
   - 若不想包含，可在 `.gitignore` 中添加 `*.pth`。

---

## 许可证

本项目采用 **MIT 许可证**，详情见 `LICENSE` 文件。

---

## 贡献

欢迎提交 Issue 和 Pull Request。

---

## 参考

- [PyTorch 官方文档](https://pytorch.org/)
- [MNIST 数据库](http://yann.lecun.com/exdb/mnist/)
