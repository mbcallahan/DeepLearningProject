import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class FashionMNIST_RGB(Dataset):
    """Source: (3x28x28 tensor, label)"""
    def __init__(self, base_fmnist_pil):
        self.base = base_fmnist_pil  # returns (PIL, label)
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        return self.tf(img), y

class FashionMNISTM_ColorDigit(Dataset):
    """
    Target: generator:
      - CIFAR-10 texture background patch
      - Fashion item ink is colorized (random RGB)
      - blended via soft mask
    Returns: (3x28x28 tensor, same label)
    """
    def __init__(self, base_fmnist_pil, bg_source_pil, seed=0, mask_power=1.4):
        self.base = base_fmnist_pil
        self.bg = bg_source_pil
        self.rng = random.Random(seed)
        self.mask_power = float(mask_power)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def _random_bg_patch(self):
        idx = self.rng.randrange(len(self.bg))
        img, _ = self.bg[idx]
        img = img.convert("RGB")  # 32x32

        x0 = self.rng.randrange(0, img.size[0] - 28 + 1)
        y0 = self.rng.randrange(0, img.size[1] - 28 + 1)
        return img.crop((x0, y0, x0 + 28, y0 + 28))

    def _random_digit_color(self):
        # avoid very dark digit colors
        return np.array([
            self.rng.randint(80, 255),
            self.rng.randint(80, 255),
            self.rng.randint(80, 255),
        ], dtype=np.float32) / 255.0

    def __getitem__(self, i):
        digit, y = self.base[i]  # PIL L
        digit = digit.convert("L").resize((28, 28), Image.BILINEAR)
        bg = self._random_bg_patch()  # PIL RGB 28x28

        digit_np = np.array(digit).astype(np.float32) / 255.0      # HxW
        mask = np.clip(digit_np, 0.0, 1.0) ** self.mask_power      # HxW
        mask = mask[..., None]                                     # HxWx1

        bg_np = np.array(bg).astype(np.float32) / 255.0            # HxWx3
        color = self._random_digit_color()[None, None, :]          # 1x1x3

        # colored ink intensity follows digit brightness
        digit_rgb = color * digit_np[..., None]                    # HxWx3

        out = bg_np * (1.0 - mask) + digit_rgb * mask              # HxWx3
        out = np.clip(out, 0.0, 1.0)

        out_pil = Image.fromarray((out * 255).astype(np.uint8), mode="RGB")
        x = self.to_tensor(out_pil)  # 3x28x28
        return x, y
class FashionMNISTM_EvenBackground(Dataset):
    """
    Target: generator:
      - Background is even RGB
      - Fashion item ink is colorized (random RGB)
      - blended via soft mask
    Returns: (3x28x28 tensor, same label)
    """
    def __init__(self, base_fmnist_pil, seed=0, mask_power=1.4):
        self.base = base_fmnist_pil
    
        self.rng = random.Random(seed)
        self.mask_power = float(mask_power)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def _random_bg_patch(self):
        r = self.rng.randrange(255)
        g = self.rng.randrange(255)
        b = self.rng.randrange(255)
        
        img =Image.new('RGB',(28,28),(r,g,b))   # even color background

        return img

    def _random_digit_color(self):
        # avoid very dark digit colors
        return np.array([
            self.rng.randint(80, 255),
            self.rng.randint(80, 255),
            self.rng.randint(80, 255),
        ], dtype=np.float32) / 255.0

    def __getitem__(self, i):
        digit, y = self.base[i]  # PIL L
        digit = digit.convert("L").resize((28, 28), Image.BILINEAR)
        bg = self._random_bg_patch()  # PIL RGB 28x28

        digit_np = np.array(digit).astype(np.float32) / 255.0      # HxW
        mask = np.clip(digit_np, 0.0, 1.0) ** self.mask_power      # HxW
        mask = mask[..., None]                                     # HxWx1

        bg_np = np.array(bg).astype(np.float32) / 255.0            # HxWx3
        color = self._random_digit_color()[None, None, :]          # 1x1x3

        # colored ink intensity follows digit brightness
        digit_rgb = color * digit_np[..., None]                    # HxWx3

        out = bg_np * (1.0 - mask) + digit_rgb * mask              # HxWx3
        out = np.clip(out, 0.0, 1.0)

        out_pil = Image.fromarray((out * 255).astype(np.uint8), mode="RGB")
        x = self.to_tensor(out_pil)  # 3x28x28
        return x, y

if __name__=="__main__":
    

    """Fashion Dataset"""  ###### comment if you want to use Digits
    # Load base datasets
    fmnist_train_pil = datasets.FashionMNIST("./data", train=True, download=False, transform=None)
    fmnist_test_pil  = datasets.FashionMNIST("./data", train=False, download=False, transform=None)
    # Labels/names come from the data file:
    CLASS_NAMES = fmnist_train_pil.classes
    print("CLASS_NAMES:", CLASS_NAMES)
    # Background textures
    cifar_train_pil = datasets.CIFAR10("./data", train=True, download=False, transform=None)
     # Source and Target datasets
    src_train = FashionMNIST_RGB(fmnist_train_pil)
    src_test  = FashionMNIST_RGB(fmnist_test_pil)
    tgt_train = FashionMNISTM_EvenBackground(fmnist_train_pil, seed=0, mask_power=1.4)
    tgt_test  = FashionMNISTM_EvenBackground(fmnist_test_pil, seed=1, mask_power=1.4)
    print("Source train/test:", len(src_train), len(src_test))
    print("Target train/test:", len(tgt_train), len(tgt_test))


    """Digit Dataset"""   ######comment out if you want to use Digits
    # mnist_train_pil = datasets.MNIST("./data", train=True, download=True, transform=None)
    # mnist_test_pil  = datasets.MNIST("./data", train=False, download=True, transform=None)
    # CLASS_NAMES = [str(i) for i in range(10)]  # digits 0-9
    # cifar_train_pil = datasets.CIFAR10("./data", train=True, download=True, transform=None)
    # src_train = FashionMNIST_RGB(mnist_train_pil)
    # src_test  = FashionMNIST_RGB(mnist_test_pil)
    # tgt_train = FashionMNISTM_ColorDigit(mnist_train_pil, cifar_train_pil, seed=0, mask_power=1.4)
    # tgt_test  = FashionMNISTM_ColorDigit(mnist_test_pil,  cifar_train_pil, seed=1, mask_power=1.4)

    batch_size = 64
    half = batch_size // 2

    src_loader = DataLoader(src_train, batch_size=half, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    tgt_loader = DataLoader(tgt_train, batch_size=half, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    src_test_loader = DataLoader(src_test, batch_size=256, shuffle=False, num_workers=2)
    tgt_test_loader = DataLoader(tgt_test, batch_size=256, shuffle=False, num_workers=2)

    
    def show_pair(i=0):
        xs, ys = src_train[i]
        xt, yt = tgt_train[i]

        xs_img = xs.permute(1, 2, 0).numpy()
        xt_img = xt.permute(1, 2, 0).numpy()

        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(xs_img)
        ax1.set_title(f"Source: {CLASS_NAMES[int(ys)]}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(xt_img)
        ax2.set_title(f"Target: {CLASS_NAMES[int(yt)]}")
        ax2.axis("off")

        plt.show()

    show_pair(0)
    show_pair(1)
    show_pair(2)


