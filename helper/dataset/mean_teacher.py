from dataset.template import TemplateDataset

class MeanTeacherDataset(TemplateDataset):
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
        
        self.csv_data = df = pd.read_csv("data/data.csv")
        
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

        path = self.csv_data.iloc[index]['img_path']       
        image = Image.open(path).convert('RGB')
            
        if self.transforms:
            pass
            
        mask = self.csv_data.iloc[index]['msk_path']
        mask = Image.open(path).convert('L')
        
        """
                
        # 's_inp', 't_inp' and 'gt' are tuples
            s_inp, t_inp, gt = self._batch_prehandle(inp, gt, True) # this should be in dataloader
            """
        
        # To change: you can add labels here
        item = {
            'img_s' : image,
            'img_t' : image,
            'msk' : mask
        } 
        
        return item
    
    def get_class_labels(self):
        return list(self.csv_data["label"])
    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        transform_list = []
        transform_list.append(ResizeCrop(self.image_size))
        transform_list.append(RandomTransforms())
        transform_list.append(ToTensor())
        transform_list.append(Normalise())
        return transform_list