from torch.utils.data import Dataset

class TemplateDataset(Dataset):
    # =============================================================================
    #
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, mode, image_size=500):
        super(TemplateDataset, self).__init__()
        
        self.mode = mode # train/val
        self.image_size = image_size
        
        self.csv_data = pd.read_csv("data/data.csv")
        
        self.transforms = transforms.Compose(self.get_transforms())
         
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

        path = self.csv_data.iloc[index]['image_path']       
        image = Image.open(path)
            
        if self.transforms:
            # apply transforms to both images
            tct = TwoCropTransform(self.transforms)
            image = tct(image)
            
        label = self.csv_data.iloc[index]['lbl']
        
        # To change: you can add labels here
        item = {
            'img' : image,
            'lbl' : label
        } 
        
        return item
    
    def get_class_labels(self):
        return list(self.csv_data["lbl"])
    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        transform_list = [
            ResizeCrop(self.image_size),
            RandomVerticalFlip(p=0.1),
            RandomHorizontalFlip(p=0.5),
            RandomAugmentations(p=0.3),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        return transform_list