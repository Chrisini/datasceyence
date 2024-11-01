from data.template import TemplateDataset

class KFoldDataset(TemplateDataset):
    # =============================================================================
    #
    # Dataset (Helper Class)
    #
    # =============================================================================

    def __init__(self, csv_data_path, dataset="train", max_k=5):
        
        self.transforms = transforms.Compose(self._get_transforms())
        self.csv_data_original = pd.read_csv(csv_data_path, delimiter=";")
        
        self.csv_data_original = self.csv_data_original.sample(frac=1, random_state=79)        

        self.k_ids = []

        parts = np.array_split(self.csv_data_original, max_k)

        for part in parts:
            #print(part)
            train_ids = self.csv_data_original[~self.csv_data_original.image_name.isin(part.image_name)]
            #adf[~adf.x1.isin(bdf.x1)]
            val_ids = part
            self.k_ids.append({"train": train_ids, "val":val_ids})

        self.dataset = dataset
        self.csv_data = self.get_next_k(k=0)

    def get_next_k(self, k=0):

        if self.dataset=="train": 
            self.csv_data = self.k_ids[k]["train"]
        elif self.dataset=="val": 
            self.csv_data = self.k_ids[k]["val"]

        
        print("------------------", self.csv_data.__len__())


    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index=index.tolist()

        image_dir = "data/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set"
        image_path = os.path.join(image_dir, self.csv_data.iloc[index]["image_name"])
        image = Image.open(image_path).convert('RGB')

        # To change: you can add labels here
        item = {
            'image'   : image,
            'grade'   : int(self.csv_data.iloc[index]["DR_grade"])
        } 

        if self.transforms:
            item = self.transforms(item)

        return item
    
    def _get_transforms(self):
        # =============================================================================
        # notes:
        # =============================================================================

        transform_list = []
        transform_list.append(ResizeCrop(image_size=512))
        transform_list.append(ToTensor())
        transform_list.append(Normalise())
        return transform_list

    