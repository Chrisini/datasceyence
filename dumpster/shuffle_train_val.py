csv_data = csv_data.sample(frac=1, random_state=19).reset_index(drop=True)

percent95 = int(len(csv_data)/100*95)

if mode == "train":
    # 95% of data
    csv_data = csv_data[0:percent95]
    print("trainset length", len(csv_data))
    print("value count:", csv_data['label'].value_counts())

else:
    # 5% of data
    csv_data = csv_data[percent95:-1]
    print("valset length",len(csv_data))
    print("value count:", csv_data['label'].value_counts())