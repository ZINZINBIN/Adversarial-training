from src.utils import generate_dataloader, get_mnist_from_sklearn

train_loader, valid_loader, test_loader = generate_dataloader(64)

sample = next(iter(train_loader))

print(sample[0].size())
print(sample[0].type())

img, label = get_mnist_from_sklearn()

print(img.shape)
print(label.shape)
print(label)