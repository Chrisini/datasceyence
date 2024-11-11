import torch
from torch.utils.data import Dataset     
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from data.template import TemplateDataLoaderWrapper, TemplateDataset
from data.transform.octa500_flatten import *
from data.transform.octa500_crop import *
from data.transform.octa500_resize import *

from PIL import Image
    

INFO = {
    "octa500": {
        "python_class": "OCTA500",
        "description": "The OCTA500 is based on optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split the dataset as needed based on a CSV file into training, validation and testset. Possible split for validation of the decentnet: 0:0:180",
        "task": "multi-class",
        "label": {
            "0": "cnv", # cnv + amd?
            "1": "dr", # dr
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
        
        # dataset
        trainset = OCTA500Dataset(mode="train", transforms=self.transforms, index_col=None, channels=1, 
                                  image_size=train_kwargs["img_size"], csv_filenames=train_kwargs["input_data_csv"], p_aug=0.5)  
        valset = OCTA500Dataset(mode="val", transforms=self.transforms, index_col=None, channels=1, 
                                image_size=train_kwargs["img_size"], csv_filenames=train_kwargs["input_data_csv"], p_aug=0.5)
        testset = OCTA500Dataset(mode="test", transforms=self.transforms, index_col=None, channels=1, 
                                 image_size=train_kwargs["img_size"], csv_filenames=train_kwargs["input_data_csv"], p_aug=0.5) 
        
        # info
        INFO['octa500']["n_samples"]["train"] = len(trainset) # should be 0
        INFO['octa500']["n_samples"]["val"] = len(valset) # should be 0
        INFO['octa500']["n_samples"]["test"] = len(testset) # should be 180
        self.info = INFO['octa500']
        model_kwargs['n_classes'] = len(self.info['label'])

        # from parent
        train_indices, val_indices, test_indices = self.get_indices(train_kwargs)
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData             
        self.log_info()
        
    def get_transforms(self, train_kwargs):
        
        # grayscale (1 channel)
        transform_list = [
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
        
        self.csv_data = []
        csv_list = []

        for i, filename in enumerate(csv_filenames):
            df = pd.read_csv(filename, delimiter=";", index_col=index_col)
            # df["dataset_type"] = [i]*len(df.index)
            csv_list.append(df)

        self.csv_data = pd.concat(csv_list, axis=0, ignore_index=False)
        self.csv_data = self.csv_data[self.csv_data["mode"].str.contains(mode)]
        
        #print("here template")
        #print(self.csv_data)
        
        print(mode)
        print(self.csv_data)
        print(len(self.csv_data.index))
        
        #if reduced_data:
        #    self.csv_data = self.csv_data.sample(frac=1).reset_index(drop=True)
        #    self.csv_data = self.csv_data.head(200)
        
        # self.transforms = torchvision.transforms.Compose(self.get_transforms())
        
    
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
        # image = Image.open(skimage.io.imread(filename)   
        img = skimage.io.imread(path)
        
        img_id_minus_one = 200 - 1
        path = self.csv_data.iloc[index]['msk_path']
        mat = scipy.io.loadmat(path)
        msk = mat["Layer"][:, img_id_minus_one]
        
        lbl = self.csv_data.iloc[index]['lbl_disease']
        
        img_id = self.csv_data.iloc[index]['img_id']
                
        if self.transforms:
            flt = octa500_flatten(img.copy(), msk.copy())
            # img, msk = flt.execute() # flattened
            img, msk, _ = octa500_crop(img, msk) # should probably save this mask too ? 
            
            
            _, msk_22 = octa500_resize(img=img, msk=msk, size=self.image_size-6)
            _, msk_24 = octa500_resize(img=img, msk=msk, size=self.image_size-4)
            _, msk_26 = octa500_resize(img=img, msk=msk, size=self.image_size-2)
            img, msk_28 = octa500_resize(img=img, msk=msk, size=self.image_size)
            
            msks = [msk_28,msk_26,msk_24,msk_22]
            
            img = Image.fromarray(img)
            img = self.transforms(img)

        lbl = torch.tensor(lbl, dtype=torch.long)
        
        img_id = torch.tensor(img_id, dtype=torch.long)
        
        msks = torch.tensor(np.array(msks))
        
        return img, lbl, msks, img_id # rewrite function in case other stuff is needed
    

            
