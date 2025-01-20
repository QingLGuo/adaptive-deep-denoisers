import os
import torch.optim as optim
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd
from PIL import Image  # Import the Image class from the PIL module
from torchvision import transforms as transforms
import torch.nn as nn
from denoisers.utils import *
from adaptive_deep_dsg_nlm import ADDSGNLM

# 自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("L")  # 转换为灰度图像

        if self.transform:
            image = self.transform(image)

        return image

# 加载训练集和测试集
train_transform = transforms.Compose([
    transforms.RandomCrop((64, 64)),  # 调整图像大小
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    # transforms.RandomCrop((180, 180)),
    transforms.ToTensor()
])

trainset = CustomDataset(root_dir="./data/Train400", transform=train_transform)
testset = CustomDataset(root_dir='./data/TestData/set12', transform=test_transform)
# testset = CustomDataset(root_dir='./data/Train400', transform=test_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


def add_noise(images, noise_factor=10/255):
    noise = noise_factor * torch.randn_like(images)
    images_noisy = torch.clamp(images + noise, 0., 1.)
    return images_noisy

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ADDSGNLM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for name,param in model.named_parameters():
        print(f"Parameter name: {name}, Requires grad: {param.requires_grad}")
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data
            inputs = inputs.to(device)
            inputs2 = add_noise(inputs)
            inputs2 = inputs2.to(device)

            optimizer.zero_grad()
            outputs = model(inputs2)
            loss = criterion(outputs, inputs)
            print(f'loss_{i}:{loss:.8f}')

            loss.backward()

            # loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f'{name}:Gradient = {param.grad.abs().mean().item()}')

            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")
        torch.save(model.state_dict(), f'ADDSGNLM_noise10_{epoch}.pth')

    torch.save(model.state_dict(), 'ADDSGNLM_noise10.pth')


def test_model_with_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ADDSGNLM().to(device)
    model.load_state_dict(torch.load('ADDSGNLM_noise10.pth'))

    model.eval()
    result_dict = {"ImageName": [], "PSNR": [], "SSIM": []}
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs = data  # Separate inputs and labels
            inputs2 = add_noise(inputs)
            noise_img_np = inputs2.squeeze().cpu().numpy()
            noise_img_np = Image.fromarray((noise_img_np * 255).astype(np.uint8))
            noise_img_np.save(f'./traind_model/noise_10_img_{i+1}.png')
            inputs2 = inputs2.to(device)
            outputs = model(inputs2)
            outputs = torch.clamp(outputs, 0., 1.)

            for j in range(len(outputs)):
                output_image = outputs[j].cpu().numpy()
                target_image = inputs[j].cpu().numpy()
                output_image = np.squeeze(output_image)
                target_image = np.squeeze(target_image)

                output_image = (output_image * 255).astype("uint8")
                target_image = (target_image * 255).astype("uint8")

                psnr = peak_signal_noise_ratio(target_image, output_image)
                ssim = structural_similarity(target_image, output_image, multichannel=False, gaussian_weights=True, use_sample_covariance=False, sigma=1.5, K1=0.01, K2=0.03)

                output_image_path = os.path.join("./traind_model", f"output_denoised__{i * len(outputs) + j}.png")
                save_image(outputs[j], output_image_path)

                result_dict["ImageName"].append(f"output_10_{i * len(outputs) + j}.png")
                result_dict["PSNR"].append(psnr)
                result_dict["SSIM"].append(ssim)

        df = pd.DataFrame(result_dict)
        df.to_csv("ADDSGNLM_denoised10.csv", index=False)

if __name__ == "__main__":

    train_model()

    test_model_with_metrics()