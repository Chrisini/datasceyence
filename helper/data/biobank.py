import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class DataBiobank(Dataset):
    def __init__(self, csv_file, root_dirs, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dirs = root_dirs
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        directory = self.data.iloc[idx, 2]
        img_path = os.path.join(self.root_dirs[directory], img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Example usage
csv_file = 'path/to/your/labels.csv'
root_dirs = {'dir1': 'path/to/dir1', 'dir2': 'path/to/dir2', 'dir3': 'path/to/dir3'}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = CustomImageDataset(csv_file=csv_file, root_dirs=root_dirs, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Example to iterate through the dataloader
for images, labels in dataloader:
    # Do something with images and labels
    pass