from dataset.template import *


class MeanTeacherTrainDataset(TemplateDataset):
    # =============================================================================
    #
    # val needs different transforms!!!! todo
    #
    # =============================================================================
    
    def __init__(self, mode="train", channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"], reduced_data=False, i2itype=None):
        super().__init__(mode=mode, index_col=None, channels=channels, image_size=image_size, csv_filenames=csv_filenames, reduced_data=reduced_data)
        
        self.i2itype = i2itype

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

        i_path = self.csv_data.iloc[index]['img_path']    
                
        if self.channels == 1:
            # image = Image.open(i_path).convert('L')
            image = skimage.io.imread(i_path, as_gray=True)
        else:
            # image = Image.open(i_path).convert('RGB')
            image = skimage.io.imread(i_path, as_gray=False)
                       
        if 'msk_path' in self.csv_data.iloc[index].keys():
            if self.csv_data.iloc[index]['msk_path'] is not np.nan:
                m_path = self.csv_data.iloc[index]['msk_path']
                #print(i_path)
                #print(m_path)
                # mask = Image.open(m_path).convert('L')
                mask = skimage.io.imread(i_path, as_gray=True)
                has_mask = True
            else: 
                mask = None
                has_mask = False
        else: 
            mask = None
            has_mask = False
                
        weight = self.csv_data.iloc[index]['weight']
        dataset_type = self.csv_data.iloc[index]['dataset_type']
        should_crop = self.csv_data.iloc[index]['should_crop']
        
        # has_mask = self.csv_data.iloc[index]['has_mask']
        
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image,
            'msk' : mask,
            "weight" : weight,
            "mbs_class" : dataset_type, # mixed batch sampler, class for data imbalance handling
            "has_mask" : has_mask, # duplicate with labelled
            "should_crop" : should_crop # for fundus images, crop around the mask
        } 
        
        if self.transforms:
            item = self.transforms(item)
            
        if item["msk"] is None:
            item["msk"] = torch.zeros(1, self.image_size, self.image_size)
            # item["msk"] = torch.empty((0), dtype=torch.float32)
        
        return item
    
    def get_mbs_labels(self):
        # =============================================================================
        # notes:
        #    mixed batch sampler based on the dataset: adam, plex, palm, refuge, ...  
        # =============================================================================
        return list(self.csv_data["dataset_type"])
    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        paths = self.csv_data.loc[df['mask_path'] == None]["img_path"]
        i_path = random.choice(paths)
        if self.channels == 1:
            # image = Image.open(i_path).convert('L')
            tgt_img = skimage.io.imread(i_path, as_gray=True)
        else:
            # image = Image.open(i_path).convert('RGB')
            tgt_img = skimage.io.imread(i_path, as_gray=False)
        
        transform_list = [
            MaskCrop(),
            FourierDomainAdapTransform(self.i2itype, tgt_img),
            ToPillow(),
            ResizeCrop(self.image_size),
            RandomVerticalFlip(p=0.1),
            RandomHorizontalFlip(p=0.5),
            RandomAugmentations(p=0.3),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list
    
    
class MeanTeacherValDataset(TemplateDataset):
    # =============================================================================
    #
    # val needs different transforms!!!! todo
    #
    # =============================================================================

    def __init__(self, mode="val", channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"], reduced_data=False):
        super().__init__(mode=mode, index_col=None, channels=channels, image_size=image_size, csv_filenames=csv_filenames, reduced_data=reduced_data)
        
         
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

        i_path = self.csv_data.iloc[index]['img_path']    
        if self.channels == 1:
            image = Image.open(i_path).convert('L')
        else:
            image = Image.open(i_path).convert('RGB')
           
        m_path = self.csv_data.iloc[index]['msk_path']
        mask = Image.open(m_path).convert('L')
  
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image,
            'msk' : mask,
        } 
        
        if self.transforms:
            item = self.transforms(item)
        
        return item
    

    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        transform_list = [
            ToPillow(),
            ResizeCrop(self.image_size),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list
    
    
class MeanTeacherCirDataset(TemplateDataset):
    # =============================================================================
    #
    # val needs different transforms!!!! todo
    #
    # =============================================================================

    def __init__(self, channels=1, image_size=500, csv_filenames=["datasceyence-master2/data_prep/mt_data_cirrus.csv"], reduced_data=False):
        
        self.image_size = image_size
                
        self.channels=channels
        
        csv_list = []

        for i, filename in enumerate(csv_filenames):
            df = pd.read_csv(filename, delimiter=";", index_col=None)
            df["dataset_type"] = [i]*len(df.index)
            csv_list.append(df)

        self.csv_data = pd.concat(csv_list, axis=0, ignore_index=False)
        
        self.transforms = torchvision.transforms.Compose(self.get_transforms())
        
         
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

        i_path = self.csv_data.iloc[index]['img_path']    
                
        if self.channels == 1:
            image = Image.open(i_path).convert('L')
        else:
            image = Image.open(i_path).convert('RGB')
           
        if 'msk_path' in self.csv_data.iloc[index].keys():
            if self.csv_data.iloc[index]['msk_path'] is not np.nan:
                m_path = self.csv_data.iloc[index]['msk_path']
                #print(i_path)
                #print(m_path)
                mask = Image.open(m_path).convert('L')
                has_mask = True
            else: 
                mask = None
                has_mask = False
        else: 
            mask = None
            has_mask = False
                
        weight = self.csv_data.iloc[index]['weight']
        
        dataset_type = self.csv_data.iloc[index]['dataset_type']
        
        # has_mask = self.csv_data.iloc[index]['has_mask']
        
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image, # img_s
            # 'img_t' : image,
            'msk' : mask,
            "weight" : weight,
            "mbs_class" : dataset_type, # mixed batch sampler, class for data imbalance handling
            "has_mask" : has_mask, # duplicate with labelled
        } 
        
        if self.transforms:
            item = self.transforms(item)
            
        if item["msk"] is None:
            item["msk"] = torch.zeros(1, self.image_size, self.image_size)
            # item["msk"] = torch.empty((0), dtype=torch.float32)
        
        return item
    
    

    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        transform_list = [
            ToPillow(),
            ResizeCrop(self.image_size),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list