# =============================================================================
# i don't know whether this works
# =============================================================================
def visualise_saliency(self, eye, configs, iteration, image_id):
    # =============================================================================
    # not sure it learns with backprop
    # not good in "per head", since I have a output=eye_model(image) here
    # targets are wrong
    # Attention map using gradients
    # neeeds grad, don't use "with torch.no_grad():" before
    # source: 
    #    https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80
    #    https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
    #    https://github.com/osu-xai/pytorch-saliency/blob/master/saliency/vanilla/saliency.py
    #    https://github.com/osu-xai/pytorch-saliency/blob/master/saliency/saliency.py
    # ============================================================================= 

    #for head_key in configs.head_input_list:            
    for meta_reader_key, meta_reader_value in eye.items():
        # ignore, if image key
        if meta_reader_key == 'image':
            continue
        if "phase" in meta_reader_key.lower() and "phase" not in self.head_name.lower():
            continue
        if "lat" in meta_reader_key.lower() and "lat" not in self.head_name.lower():
            continue
        if "norm" in meta_reader_key.lower() and "reg" not in self.head_name.lower():
            continue

        print("NEEEEEEEEEEW")
        print(image_id)
        print(meta_reader_key)
        print(iteration)
        print(self.head_name)

        # load model weights new each time an image is put through?? probably not necessary        
        eye_tmp_saliency_model = EyeNet(configs.model_resnet, configs.heads).to(configs.device)
        eye_tmp_saliency_model.load_state_dict(torch.load("results/saliency/tmp_saliency_model.ckpt"))
        optimizer = torch.optim.Adam(eye_tmp_saliency_model.parameters(), lr=configs.learning_rate)
        eye_tmp_saliency_model.eval()        

        # get image, reshape, to device, requires grad
        image = eye["image"][0]
        image = image.reshape(1, 3, configs.image_size, configs.image_size)
        image = image.to(configs.device)
        image.requires_grad = True

        # model to zero grad
        eye_tmp_saliency_model.zero_grad()

        ground_truth = eye[meta_reader_key]


        # model output
        model_output = eye_tmp_saliency_model(image)

        #print("o", model_output)
        #print("m", model_output[self.head_name])
        #print("gt", ground_truth)

        # one hot encoding: set all to zero

        #print("g1", grad_outputs)
        if "Default_BinClass_Lat_Head" in self.head_name:

            grad_outputs = torch.zeros_like(model_output[self.head_name])
            grad_outputs[:, 0] = 1  # maybe todo

            model_output[self.head_name].backward(gradient = grad_outputs)

            # [1]
            # maybe wrong

            saliency_image = torch.sigmoid(image.grad.data.abs()) > 0.5 # = torch.max(image.grad.data.abs(),dim=1)
            saliency_image = saliency_image.reshape(configs.image_size, configs.image_size).cpu().detach()

        elif "Default_Reg_Phase_Head" in self.head_name: # this won't work at all ...
            grad_outputs = torch.zeros_like(model_output[self.head_name])
            grad_outputs[:, 0] = 0 # ground_truth.data, would need to be class activation
            # todo
        else:
            # only works for multiple output neurons # todo
            # one hot encoding
            grad_outputs = torch.zeros_like(model_output[self.head_name])
            grad_outputs[:, ground_truth] = 1 # if my class == 3 then I get 0 0 0 1 0


            model_output[self.head_name].backward(gradient = grad_outputs)


            # [1]
            saliency_image, _ = torch.max(image.grad.data.abs(),dim=1)
            saliency_image = saliency_image.reshape(configs.image_size, configs.image_size).cpu().detach()


        # [0]
        image.requires_grad = False
        original_image=image.squeeze().cpu().detach().permute(1, 2, 0)


        # [2]
        im = image.grad.clone()[0] * image
        im = im.squeeze().cpu().detach().permute(1, 2, 0)
        gradient_image = (im - im.min()) / (im.max() - im.min())

        # Visualize image, saliency map, normalised gradient*image
        fig, ax = plt.subplots(1, 3)
        # original image
        ax[0].imshow(original_image)
        ax[0].axis('off')
        fig.suptitle(f'{self.head_name} {meta_reader_key}')
        # saliency
        ax[1].imshow(saliency_image)
        ax[1].axis('off')
        # normalised, grad*image
        ax[2].imshow(gradient_image)
        ax[2].axis('off')

        plt.tight_layout()
        plt.show()

        if not os.path.exists("results/saliency/"):
            os.makedirs("results/saliency/")
        fig.savefig(f'results/saliency/{configs.name}_it{iteration}_{self.head_name}_{meta_reader_key}_{image_id}.png', bbox_inches='tight')

        plt.close()

        eye_tmp_saliency_model = eye_tmp_saliency_model.cpu()
        del eye_tmp_saliency_model
        del model_output
        del optimizer





