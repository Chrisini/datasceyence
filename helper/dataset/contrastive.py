from template import TemplateDataset

class ContrastiveDataset(TemplateDataset):
    # =============================================================================
    #
    #
    # =============================================================================
    
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
        
        item = {
            'img' : image,
            'lbl' : label
        } 
        
        return item
    
    