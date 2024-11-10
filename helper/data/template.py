from data.transform.transform import *
from data.transform.image2image import *

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms

INFO = {
    "template": {
        "python_class": "template",
        "description": "The template is based on",
        #"url": "https://zenodo.org/records/10519652/files/octmnist.npz?download=1", # -> 28
        #"MD5": "c68d92d5b585d8d81f7112f81e2d0842", # -> 28
        #"url_64": "https://zenodo.org/records/10519652/files/octmnist_64.npz?download=1",
        #"MD5_64": "e229e9440236b774d9f0dfef9d07bdaf",
        #"url_128": "https://zenodo.org/records/10519652/files/octmnist_128.npz?download=1",
        #"MD5_128": "0a97e76651ace45c5d943ee3f65b63ae",
        #"url_224": "https://zenodo.org/records/10519652/files/octmnist_224.npz?download=1",
        #"MD5_224": "abc493b6d529d5de7569faaef2773ba3", # -> other sizes
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization", # cnv + amd?
            "1": "diabetic macular edema", # dr
            "2": "drusen", # amd?
            "3": "normal", # normal
        },
        "n_channels": 1,
        "n_samples": {"train": 0, "val": 0, "test": 0},
        "license": "CC BY 4.0",
    }
}

class TemplateDataLoaderWrapper():
    # =============================================================================
    #
    # call this
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, train_kwargs, model_kwargs):
        super(TemplateData, self).__init__()
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        # dataset
        trainset = None
        valset = None
        testset = None
        
        # info
        self.info = INFO['template']
        model_kwargs['n_classes'] = len(self.info['label'])

        # from parent
        train_indices, val_indices, test_indices = self.get_indices(train_kwargs)
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData             
        self.log_info()
        
    def get_indices(self, train_kwargs): # todo, make this random!!!!
        # indices for splitting and/or reducing data
        if train_kwargs["train_size"] == -1: # all
            train_indices = range(self.info["n_samples"]["train"])
        else: # total number
            train_indices = range(train_kwargs["train_size"])
            
        if train_kwargs["val_size"] == -1: # all
            val_indices = range(self.info["n_samples"]["val"])
        else:# total number
            val_indices = range(train_kwargs["val_size"])
            
        if train_kwargs["test_size"] == -1: # all
            test_indices = range(self.info["n_samples"]["test"])
        else: # total number
            test_indices = range(train_kwargs["test_size"])
        
        return train_indices, val_indices, test_indices
    
    def log_info(self):
        
        print("********** DECENT INFO: DataLoader infos **********")
        
        for value, key in self.info.items():
            print(value, ":", key)
            
        if 'self.train_dataloader' in locals():
            print("train_dataloader", len(self.train_dataloader.dataset))
        if 'self.val_dataloader' in locals():
            print("val_dataloader", len(self.val_dataloader.dataset))
        if 'self.test_dataloader' in locals():
            print("test_dataloader", len(self.test_dataloader.dataset))
        
    def set_data(self, train_indices, val_indices, test_indices, trainset, valset, testset, train_kwargs):
        # do not change this. call in init
        
        # train subset
        if len(train_indices) > 0:
            train_subset = torch.utils.data.Subset(trainset, train_indices)
            self.train_dataloader = torch.utils.data.DataLoader(train_subset, 
                                                           shuffle=True, 
                                                           batch_size=train_kwargs["batch_size"], 
                                                           num_workers=train_kwargs["num_workers"])
        # val subset
        if len(val_indices) > 0:
            val_subset = torch.utils.data.Subset(valset, val_indices)
            self.val_dataloader = torch.utils.data.DataLoader(val_subset, 
                                                         shuffle=False, 
                                                         batch_size=train_kwargs["batch_size"], 
                                                         num_workers=train_kwargs["num_workers"]
                                                         # , persistent_workers=True
                                                         )
            
        # test subset
        if len(test_indices) > 0:
            testset = torch.utils.data.Subset(testset, test_indices)
            # test dataloader with batch size
            self.test_dataloader = torch.utils.data.DataLoader(testset, 
                                                         shuffle=False, 
                                                         batch_size=train_kwargs["batch_size"], 
                                                         num_workers=train_kwargs["num_workers"]
                                                         # , persistent_workers=True
                                                        )    

            # same as test dataloader, but batch size = 1
            self.xai_dataloader = torch.utils.data.DataLoader(testset, 
                                                         shuffle=False, 
                                                         batch_size=1, 
                                                         num_workers=train_kwargs["num_workers"]
                                                         # , persistent_workers=True
                                                        )    
       
            
        
    
    def get_transforms(self, train_kwargs):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        # gray
        transform_list = [
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # rgb to gray
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # rgb
        transform_list = [
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        
        # gray to rgb
        transform_list = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        
        # multiple
        transform_list = [
            ResizeCrop(self.image_size),
            RandomVerticalFlip(p=0.1),
            RandomHorizontalFlip(p=0.5),
            RandomAugmentationsSoft(p=self.p_aug),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        return transform_list


class TemplateDataset(Dataset):
    # =============================================================================
    #
    # use this to feed the Data
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, mode="train", index_col=None, channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"], p_aug=0.5, reduced_data=False):
        super(TemplateDataset, self).__init__()
        
        self.mode = mode # train/val
        self.image_size = image_size
        
        #self.p_aug = p_aug
        
        self.channels=channels
        
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
        return len(self.csv_data.index)
    
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

        path = self.csv_data.iloc[index]['image_path']    
        
        image = Image.open(path)            
        label = self.csv_data.iloc[index]['lbl']
        
        return image, label # rewrite function in case other stuff is needed
    
    
    def getitem_alternative(self, index):
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

        path = self.csv_data.iloc[index]['image_path']    
        
        image = Image.open(path)            
        label = self.csv_data.iloc[index]['lbl']
        
        return image, label, mask


    
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