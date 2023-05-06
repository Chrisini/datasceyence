from dataset.template import TemplateDataset


class ContrativeClustersDataset(TemplateDataset):
    # =============================================================================
    #
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, mode, channels, image_size=500, csv_filenames=["decentnet/results/tmp/masks_info_label.csv"], ci_concept=0, concepts_path="C:/Users/Prinzessin/projects/decentnet/data/tmp/concepts"):
        super().__init__(mode, channels, image_size, csv_filenames)
        
        # class speficic
        
        print(self.csv_data)
        
        self.ci_concept = ci_concept
        self.concepts_path = concepts_path  
        self.set_dataset()
         
    def __len__(self):
        return len(self.csv_data)

    def get_class_labels(self):
        return list(self.csv_data["lbl"])
    
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
            
        
        # print(image)
            
        label = self.csv_data.iloc[index]['label']
        
        # To change: you can add labels here
        item = {
            'img' : image,
            'lbl' : label
        } 
        
        return item
    
    def set_dataset(self):
        # we need a script for this
        
        image_path = []
        label = []
        
        # iterate over files in concepts path directory
        # won't work since in other dirs now
        for ci_dir in os.listdir(self.concepts_path):
            this_concept_path = os.path.join(self.concepts_path, ci_dir)
            for filename in os.listdir(this_concept_path):
                # cluster_i is 1, rest is 0
                if filename.split("_")[1] == self.ci_concept:
                    label.append(1) # positive class
                    image_path.append(os.path.join(this_concept_path, filename))
                else:
                    label.append(0) # negative class
                    image_path.append(os.path.join(this_concept_path, filename))
        
        dict = {"image_path" : image_path, "label" : label}
        self.csv_data = pd.DataFrame(dict)
        



class ContrastiveDataset_old(TemplateDataset):
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
    
    