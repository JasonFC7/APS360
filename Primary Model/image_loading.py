from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
import os
import torch

def image_loader(paths, batch_size = 16):        # paths = [train_path, val_path, test_path]
    transform = transforms.Compose([transforms.Resize((176, 208)), transforms.ToTensor()])
    
    train_dataset = GifImageFolder(paths[0], transform)
    val_dataset = GifImageFolder(paths[1], transform)
    test_dataset = GifImageFolder(paths[2], transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    print("Finished Loading")
    return train_loader, val_loader, test_loader

class GifImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(root)
        self.samples = self.make_dataset(root, self.class_to_idx)

    def find_classes(self, root):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self, root, class_to_idx):
        samples = []
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                for root_2, _, fnames in sorted(os.walk(class_dir)):
                    for fname in sorted(fnames):
                        if fname.endswith('.gif'):
                            path = os.path.join(root_2, fname)
                            item = (path, class_to_idx[class_name])
                            samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_idx = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB') 
            if self.transform:
                img = self.transform(img)
            label = torch.tensor(class_idx)
        return img, label