from visualisation.hook import Hook

import matplotlib.pyplot as plt
import numpy as np

import torch

from PIL import Image


class DeepDream():
    # =============================================================================
    # https://github.com/juanigp/Pytorch-Deep-Dream/blob/master/Deep_Dream.ipynb
    # https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    # https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
    # =============================================================================

    def __init__(self, model, layer, device="cpu", ckpt_net_path=None, iterations=200, lr=1):
        # =============================================================================
        # Initialise iter, lr, model, layer
        # =============================================================================

        # settings for dreams
        self.iterations=iterations
        self.lr=lr
        self.device = device

        # model
        if ckpt_net_path is not None:
            model.load_state_dict(torch.load(ckpt_net_path)["model"]) # 'dir/decentnet_epoch_19_0.3627.ckpt'
        self.model = model.eval()
        
        # the (conv) layer to be visualised
        self.layer = layer
        print("Layer:", self.layer)
    
    def _get_gradients(self, img):     
        # =============================================================================
        # Gradient calculations from output channels of the target layer  
        # =============================================================================
        img = img.unsqueeze(0).to(self.device)
        img.requires_grad = True
        self.model.zero_grad()
        hook = Hook(self.layer)
        _ = self.model(img)
        loss = hook.input[0].norm()
        loss.backward()
        return img.grad.data.squeeze()

    def run(self, img_tensor):
        # =============================================================================
        # Torch dreams
        # =============================================================================
        self.img_tensor = img_tensor

        for i in range(self.iterations):
          gradients = self._get_gradients(self.img_tensor).data
          self.img_tensor.data = self.img_tensor.data + self.lr * gradients
        
        # make pillow image
        img_out = self.img_tensor.detach().cpu()
        img_out = img_out.numpy().transpose(1,2,0)
        img_out = np.clip(img_out, 0, 1)
        img_out = Image.fromarray(np.uint8(img_out * 255))
        self.img_pil = img_out

    def plot(self):
        # =============================================================================
        # Plot torch dream
        # =============================================================================
        # img = img.resize(orig_size)
        fig = plt.figure(figsize = (10 , 10))
        plt.imshow(self.img_pil)

