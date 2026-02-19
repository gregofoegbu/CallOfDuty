from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

dataset = MNIST(root="data", train=True, download=True, transform=transform)

# Filter digits 1–8
dataset.data = dataset.data[(dataset.targets >= 1) & (dataset.targets <= 8)]
dataset.targets = dataset.targets[(dataset.targets >= 1) & (dataset.targets <= 8)]