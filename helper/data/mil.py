from data.template import *
import skimage.io
import skimage.util


class MilTrainDataset(TemplateDataset):
    # =============================================================================
    #
    #
    # =============================================================================
    
    def __init__(self, mode="train", channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"], reduced_data=False, i2itype=None):
        super().__init__(mode=mode, index_col=None, channels=channels, image_size=image_size, csv_filenames=csv_filenames, reduced_data=reduced_data)
        
        self.i2itype = i2itype

    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        # =============================================================================
        # Describe what is going on
        # parameters:
        #    parameter1: e.g. hidden vector of shape [bsz, n_views, ...].
        #    parameter2: e.g. ground truth of shape [bsz].
        # returns:
        #    parameter2: e.g. a loss scalar.
        # saves:
        #    collector of data within a class 
        # writes:
        #    csv file, png images, ...
        # notes:
        #    Whatever comes into your mind
        # sources:
        #    https...
        # =============================================================================
        
        if torch.is_tensor(index):
            index=index.tolist()

        img_path = self.csv_data.iloc[index]['img_path']    
                
        if self.channels == 1:
            # image = Image.open(img_path).convert('L')
            if ".tif" in img_path:
                image = skimage.io.imread(img_path, as_gray=True, plugin='tifffile')
            else:
                image = skimage.io.imread(img_path, as_gray=True)
                image = skimage.util.img_as_ubyte(image, force_copy=False)
                
            #print(image.shape)
            #print(image.dtype)
                
        else:
            # image = Image.open(img_path).convert('RGB')
            if ".tif" in img_path:
                image = skimage.io.imread(img_path, as_gray=False, plugin='tifffile')
            else:
                image = skimage.io.imread(img_path, as_gray=False)
                       
        if 'msk_path' in self.csv_data.iloc[index].keys():
            if self.csv_data.iloc[index]['msk_path'] is not np.nan:
                m_path = self.csv_data.iloc[index]['msk_path']
                #print(img_path)
                #print(m_path)
                # mask = Image.open(m_path).convert('L')
                mask = skimage.io.imread(m_path, as_gray=True)
                has_mask = True
            else: 
                mask = None
                has_mask = False
        else: 
            mask = None
            has_mask = False
                
        weight = self.csv_data.iloc[index]['weight']
        dataset_type = self.csv_data.iloc[index]['dataset_type']
        mask_crop = self.csv_data.iloc[index]['mask_crop']
        
        # has_mask = self.csv_data.iloc[index]['has_mask']
        
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image,
            'msk' : mask,
            "weight" : weight,
            "mbs_class" : dataset_type, # mixed batch sampler, class for data imbalance handling
            "has_mask" : has_mask, # duplicate with labelled
            "mask_crop" : mask_crop # for fundus images, crop around the roi of the mask mask
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
        # =============================================================================

        """
        paths = self.csv_data.loc[self.csv_data['mask_path'] == None]["img_path"]
        img_path = random.choice(paths)
        if self.channels == 1:
            # image = Image.open(img_path).convert('L')
            tgt_img = skimage.io.imread(img_path, as_gray=True)
        else:
            # image = Image.open(img_path).convert('RGB')
            tgt_img = skimage.io.imread(img_path, as_gray=False)
          
        """


        # for image to image translation
        tgt_paths = self.csv_data.loc[self.csv_data['msk_path'].isna()]["img_path"]
        
        # training transforms
        transform_list = [
            MaskCrop(image_size=self.image_size),
            FourierDomainAdapTransform(tgt_paths=tgt_paths, channels=self.channels, image_size=self.image_size),
            ToPillow(),
            ResizeCrop(image_size=self.image_size),
            RandomVerticalFlip(p=0.1),
            RandomHorizontalFlip(p=0.5),
            RandomAugmentations(p=0.3),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list
    
    
class MilValDataset(TemplateDataset):
    # =============================================================================
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
        #       mask label
        # notes:
        # =============================================================================
        
        if torch.is_tensor(index):
            index=index.tolist()

        img_path = self.csv_data.iloc[index]['img_path']    
        if self.channels == 1:
            image = skimage.io.imread(img_path, as_gray=True, plugin='tifffile')
        else:
            image = Image.open(img_path).convert('RGB')
           
        m_path = self.csv_data.iloc[index]['msk_path']
        mask = skimage.io.imread(m_path, as_gray=True)
        has_mask = True
  
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image,
            'msk' : mask,
            'has_mask' : has_mask
        } 
        
        if self.transforms:
            item = self.transforms(item)
        
        return item
    

    
    def get_transforms(self):
        # =============================================================================
        # notes:
        # =============================================================================
        filename = r'datasceyence/data_prep/mt_data_cirrus.csv'
        self.tgt_csv = pd.read_csv(filename, delimiter=";", index_col=None)        
        tgt_paths = self.tgt_csv.loc[self.tgt_csv['msk_path'].isna()]["img_path"] # gotta fix this
        
        transform_list = [
            FourierDomainAdapTransform(tgt_paths=tgt_paths, channels=self.channels, image_size=self.image_size),
            ToPillow(),
            ResizeCrop(self.image_size),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list
    
    
class MilTestDataset(TemplateDataset):
    # =============================================================================
    # Cirrus data, testset
    # =============================================================================

    def __init__(self, channels=1, image_size=500, csv_filenames=["datasceyence-master2/data_prep/test_data_cirrus.csv"], reduced_data=False):
        
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
        #       mask label
        # notes:
        # =============================================================================
        
        if torch.is_tensor(index):
            index=index.tolist()

        img_path = self.csv_data.iloc[index]['img_path']    
                
        if self.channels == 1:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
           
        if 'msk_path' in self.csv_data.iloc[index].keys():
            if self.csv_data.iloc[index]['msk_path'] is not np.nan:
                # for some reason it is suddenly np float nan ??
                m_path = self.csv_data.iloc[index]['msk_path']
                #print(img_path)
                #print(m_path)
                
                try:
                    mask = Image.open(m_path).convert('L')
                    has_mask = True
                except:
                    mask = None
                    has_mask = False
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
            'img_path' : img_path,
            'img' : image,
            'msk' : mask,
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
        # =============================================================================
        
        transform_list = [
            ResizeCrop(self.image_size),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list