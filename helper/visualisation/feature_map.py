

# =============================================================================
# Feature map visualisation using hooks       
# A high activation means a certain feature was found. 
# A feature map is called the activations of a layer after the convolutional operation.
# =============================================================================
def visualise_activation(self, eye, configs, iteration, image_id):
    activation = {}
    eye_model = configs.eye_model

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    configs.eye_model.model.layer3[0].conv1.register_forward_hook(get_activation('conv1'))

    for i, image in enumerate(eye['image']):
        image = image.to(configs.device)

        image.unsqueeze_(0)
        output = configs.eye_model(image)

        act = activation['conv1'].squeeze()

        # print("confusion", type(activation['conv1'].squeeze()))

        fig, axarr = plt.subplots(4, 4)  # act.size(0)
        plt.figure()
        # print(act.size(0))
        amount = act.shape[0]
        random_samples = random.sample(range(0, amount), 16)
        counter = 0            
        for idx in range(0, 4):
            for idx2 in range(0, 4):
                axarr[idx, idx2].axis('off')
                axarr[idx, idx2].imshow(act[random_samples[counter]].cpu())
                counter += 1

        # overwrite first image with original image        
        axarr[0,0].imshow(image.squeeze().cpu().detach().numpy().transpose(1, 2, 0))

        if not os.path.exists("results/features/"):
            os.makedirs("results/features/")
        fig.savefig(f'results/features/{configs.name}_it{iteration}_{image_id}.png', bbox_inches='tight')

        plt.close()