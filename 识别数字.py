import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

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

def get_data_loader(is_train):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    data_set = datasets.MNIST("", is_train, transform=transform, download=True)
    return DataLoader(data_set, batch_size=64, shuffle=is_train)


def evaluate(test_data, net):
    net.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            outputs = net(x)  # CNN直接接收4D输入
            pred = outputs.argmax(dim=1)
            n_correct += (pred == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

def main():
    train_data = get_data_loader(True)
    test_data = get_data_loader(False)
    net = CNNNet()
    print("Initial accuracy:", evaluate(test_data, net))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        net.train()
        for x, y in train_data:
            optimizer.zero_grad()
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        acc = evaluate(test_data, net)
        print(f"Epoch {epoch} accuracy: {acc:.4f}")

    # 显示测试图片（前4张）及其预测
    net.eval()
    with torch.no_grad():
        for n, (x, y) in enumerate(test_data):
            if n >= 4:
                break
            output = net(x[0].unsqueeze(0))   # 只取第一张图片，增加batch维度
            pred = torch.argmax(output, dim=1).item()
            plt.figure(n)
            plt.imshow(x[0].squeeze(), cmap='gray')
            plt.title(f"True: {y[0].item()}, Pred: {pred}")
    plt.show()

    torch.save(net.state_dict(), "mnist_cnn.pth")
    print("Model saved as mnist_cnn.pth")

if __name__ == "__main__":
    main()