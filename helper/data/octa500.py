import torch
from torch.utils.data import Dataset     
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io

from data.template import TemplateDataLoaderWrapper, TemplateDataset
from data.flattening import *

    

INFO = {
    "octa500": {
        "python_class": "OCTA500",
        "description": "The OCTA500 is based on optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split the dataset with a ratio of n:n:n into training, validation and testset",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization", # cnv + amd?
            "1": "diabetic retinopathy", # dr
            "2": "amd", # amd
            "3": "normal", # normal
        },
        "n_channels": 1,
        "n_samples": {"train": 0, "val": 0, "test": 0},
    }
}

path = "data_prep/data_octa_500.csv"

class DataLoaderOCTA500(TemplateDataLoaderWrapper):
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        trainset = None # OCTA500Dataset(split="train", transform=self.transforms, download=True)
        valset = None #  OCTA500Dataset(split="val", transform=self.transforms, download=True)
        testset = OCTA500Dataset(mode="test", transforms=self.transforms, index_col=None, channels=1, image_size=train_kwargs["img_size"], csv_filenames=train_kwargs["input_data_csv"], p_aug=0.5) 
        
        self.info = INFO['octa500']
        model_kwargs['n_classes'] = len(self.info['label'])

        # indices for splitting and/or reducing data
        train_indices = None # range(train_kwargs["train_size"])
        val_indices = None # range(train_kwargs["val_size"])
        test_indices = range(train_kwargs["test_size"])
        
        # from parent
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData     
        
        # from parent
        self.log_info()
        
    def get_transforms(self, train_kwargs):
        
        # grayscale (1 channel)
        transform_list = [
                          torchvision.transforms.Resize(size=train_kwargs["img_size"]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                         ]

        return transform_list
    
    
class OCTA500Dataset(TemplateDataset):
    # =============================================================================
    #
    # Dataset
    # create objects based on child class
    #
    # =============================================================================
    # split="train", transform=self.transforms, download=True
    def __init__(self, mode="test", transforms=None, index_col=None, channels=1, image_size=500, csv_filenames=["data_octa_500.csv"], p_aug=0.5):
        super(TemplateDataset, self).__init__()
        
        self.mode = mode # train/val
        self.image_size = image_size
        self.channels=channels
        self.p_aug = p_aug
        self.transforms = transforms
        
        csv_list = []

        for i, filename in enumerate(csv_filenames):
            df = pd.read_csv(filename, delimiter=";", index_col=index_col)
            df["dataset_type"] = [i]*len(df.index)
            csv_list.append(df)

        self.csv_data = pd.concat(csv_list, axis=0, ignore_index=False)
        
        #print("here template")
        #print(self.csv_data)
                
        self.csv_data = self.csv_data[self.csv_data["mode"].str.contains(mode)]
        
        #if reduced_data:
        #    self.csv_data = self.csv_data.sample(frac=1).reset_index(drop=True)
        #    self.csv_data = self.csv_data.head(200)
        
        # self.transforms = torchvision.transforms.Compose(self.get_transforms())
        
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        # =============================================================================
        # parameters:
        #   index of single image from dataloader
        # returns:
        #   dictionary "item" with:
        #       image (transformed)
        #       label
        # notes:
        # =============================================================================
        
        if torch.is_tensor(index):
            index=index.tolist()

        path = self.csv_data.iloc[index]['img_path']    
        image = Image.open(path)   
        
        
        img_id_minus_one = 200 - 1
        path = self.csv_data.iloc[index]['msk_path']
        mat = scipy.io.loadmat(path)
        mask = mat["Layer"][:, img_id_minus_one]
        
        label = self.csv_data.iloc[index]['lbl_disease']
        
        if self.transforms:
            flt = OpScanFlatten(image.copy(), mask.copy())
            image, mask = flt.execute() # flattened

        
        return image, label, mask # rewrite function in case other stuff is needed
    

            
'''
class MedMNIST(Dataset):
    flag = ...

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        as_rgb=False,
        root=DEFAULT_ROOT,
        size=None,
        mmap_mode=None,
    ):
        """
        Args:

            split (string): 'train', 'val' or 'test', required
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default: False.
            as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
            size (int, optional): The size of the returned images. If None, use MNIST-like 28. Default: None.
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.
            root (string, optional): Root directory of dataset. Default: `~/.medmnist`.

        """

        # Here, `size_flag` is blank for 28 images, and `_size` for larger images, e.g., "_64".
        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            assert size in self.available_sizes
            self.size = size
            self.size_flag = f"_{size}"

        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        if download:
            self.download()

        if not os.path.exists(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz")
        ):
            raise RuntimeError(
                "Dataset not found. " + " You can set `download=True` to download it"
            )

        npz_file = np.load(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"),
            mmap_mode=mmap_mode,
        )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split in ["train", "val", "test"]:
            self.imgs = npz_file[f"{self.split}_images"]
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def __len__(self):
        assert self.info["n_samples"][self.split] == self.imgs.shape[0]
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.size} ({self.flag}{self.size_flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info[f"url{self.size_flag}"],
                root=self.root,
                filename=f"{self.flag}{self.size_flag}.npz",
                md5=self.info[f"MD5{self.size_flag}"],
            )
        except:
            raise RuntimeError(
                f"""
                Automatic download failed! Please download {self.flag}{self.size_flag}.npz manually.
                1. [Optional] Check your network connection: 
                    Go to {HOMEPAGE} and find the Zenodo repository
                2. Download the npz file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url{self.size_flag}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5{self.size_flag}"]}
                4. Put the npz file under your MedMNIST root folder: 
                    {self.root}
                """
            )        
        

class MedMNIST2D(MedMNIST):
    available_sizes = [28, 64, 128, 224]

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="png", write_csv=True):
        from medmnist.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}{self.size_flag}.csv")
            if write_csv
            else None,
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs, n_channels=self.info["n_channels"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(
                os.path.join(
                    save_folder, f"{self.flag}{self.size_flag}_{self.split}_montage.jpg"
                )
            )

        return montage_img
    
    
'''