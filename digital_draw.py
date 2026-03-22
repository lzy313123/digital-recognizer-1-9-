import torch
import torchvision.transforms as transforms
import tkinter as tk
from PIL import Image, ImageDraw,ImageFilter
import numpy as np
import cv2

# ---------- 1. 定义与训练时相同的网络结构（CNN） ----------
class CNNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = torch.nn.Linear(64*7*7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


# ---------- 2. 加载训练好的模型 ----------
def load_model(model_path="mnist_cnn.pth"):
    model = CNNNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ---------- 3. 预处理函数：将画板图像转换为模型输入 ----------
def preprocess_image(image):
    # 1. 转为灰度并二值化（Otsu 自适应阈值）
    gray = image.convert('L')
    img_array = np.array(gray, dtype=np.uint8)
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 找到数字的边界框
    coords = cv2.findNonZero(binary)
    if coords is None:
        # 空白图像
        return torch.zeros((1, 1, 28, 28))

    x, y, w, h = cv2.boundingRect(coords)

    # 3. 裁剪出数字区域
    digit = binary[y:y+h, x:x+w]

    # 4. 如果数字太小，放大到 min_size
    min_size = 20
    if max(w, h) < min_size:
        scale = min_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        _, digit = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY)

    # 5. 形态学膨胀（使细线变粗）
    kernel = np.ones((2,2), np.uint8)
    digit = cv2.dilate(digit, kernel, iterations=1)

    # 6. 获取 digit 的最终尺寸
    h_d, w_d = digit.shape

    # 7. 如果 digit 过大（超过28），等比例缩小（防止意外）
    if h_d > 28 or w_d > 28:
        scale = 28 / max(h_d, w_d)
        new_w, new_h = int(w_d * scale), int(h_d * scale)
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        _, digit = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY)
        h_d, w_d = digit.shape

    # 8. 创建 28×28 画布并居中放置
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - h_d) // 2
    x_offset = (28 - w_d) // 2
    canvas[y_offset:y_offset+h_d, x_offset:x_offset+w_d] = digit / 255.0

    # 9. 标准化（MNIST 统计值）
    canvas = (canvas - 0.1307) / 0.3081

    # 转为张量
    return torch.tensor(canvas).unsqueeze(0).unsqueeze(0)



# ---------- 4. 创建 tkinter 画板窗口 ----------
class DrawApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("手写数字识别 - 用鼠标绘制数字")

        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height,
                                bg='white', cursor='cross')
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint_start)
        self.canvas.bind("<ButtonRelease-1>", self.paint_end)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        btn_predict = tk.Button(button_frame, text="识别", command=self.predict)
        btn_predict.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(button_frame, text="清除", command=self.clear)
        btn_clear.pack(side=tk.LEFT, padx=5)

        btn_quit = tk.Button(button_frame, text="退出", command=root.quit)
        btn_quit.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(root, text="等待绘制...", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def paint_start(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=8, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill='black', width=8)
        self.last_x = event.x
        self.last_y = event.y

    def paint_end(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="已清除")

    def predict(self):
        img_gray = self.image.convert('L')
        input_tensor = preprocess_image(img_gray)
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        self.result_label.config(text=f"识别结果: {pred}")

# ---------- 5. 主程序 ----------
def main():
    print("开始加载模型...")
    model = load_model("mnist_cnn.pth")   # 加载 CNN 模型
    print("模型加载成功！")

    root = tk.Tk()
    app = DrawApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()