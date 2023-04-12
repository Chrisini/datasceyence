def generate_concept_csv_file():
    
    # need one file for each concept 
    image_path = []
    label = []

    # iterate over files in concepts path directory
    # won't work since in other dirs now
    for ci_dir in os.listdir(concepts_path):
        this_concept_path = os.path.join(concepts_path, ci_dir)
        for filename in os.listdir(this_concept_path):
            # cluster_i is 1, rest is 0
            if filename.split("_")[1] == self.ci_concept:
                label.append(1) # positive class
                image_path.append(os.path.join(this_concept_path, filename))
            else:
                label.append(0) # negative class
                image_path.append(os.path.join(this_concept_path, filename))

    dict = {"image_path" : image_path, "label_positive" : label}
    self.csv_data = pd.DataFrame(dict)

    if False:
        self.csv_data = self.csv_data.sample(frac=1, random_state=19).reset_index(drop=True)

        percent95 = int(len(self.csv_data)/100*95)

        if self.mode == "train":
            # 95% of data
            self.csv_data = self.csv_data[0:percent95]
            print("trainset length", len(self.csv_data))
            print("value count:", self.csv_data['label'].value_counts())

        else:
            # 5% of data
            self.csv_data = self.csv_data[percent95:-1]
            print("valset length",len(self.csv_data))
            print("value count:", self.csv_data['label'].value_counts())
            
            
            
if __name__ == "__main__":
    
    run_generate_concept_csv_file = True
    
    if run_generate_concept_csv_file:
        generate_concept_csv_file()
    if True:
        pass