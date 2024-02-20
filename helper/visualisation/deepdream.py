from visualisation.hook import Hook

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

from PIL import Image


class DeepDreamLowRes():
    # =============================================================================
    # https://github.com/juanigp/Pytorch-Deep-Dream/blob/master/Deep_Dream.ipynb
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
            self.img_tensor.data = self.img_tensor.data + self.lr * gradients.data
        
        # make pillow image
        img_out = self.img_tensor.detach().cpu()
        img_out = img_out.numpy().transpose(1,2,0)
        img_out = np.clip(img_out, 0, 1)
        img_out = Image.fromarray(np.uint8(img_out * 255))
        self.img_pil = img_out

    def plot(self, size=5):
        # =============================================================================
        # Plot torch dream
        # =============================================================================
        # img = img.resize(orig_size)
        fig = plt.figure(figsize = (size , size))
        plt.imshow(self.img_pil)


        
class DeepDreamHighRes():
    # =============================================================================
    # https://github.com/juanigp/Pytorch-Deep-Dream/blob/master/Deep_Dream.ipynb
    # =============================================================================

    def __init__(self, model, layer, device="cpu", ckpt_net_path=None, out_channels=None, iterations=200, lr=1):
        # =============================================================================
        # Initialise iter, lr, model, layer
        # =============================================================================

        # settings for dreams
        self.iterations=iterations
        self.lr=lr
        self.device = device
        self.out_channels=out_channels

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
        if self.out_channels == None:
            loss = hook.output[0].norm()
        else:
            loss = hook.output[0][self.out_channels].norm()
        loss.backward()
        return img.grad.data.squeeze()

    def run(self, img_tensor):
        # =============================================================================
        # Torch dreams
        # =============================================================================
        # self.img_tensor = img_tensor
        
        img_numpy = img_tensor.detach().cpu().numpy()
        
        size = np.array(img_numpy.size)

        OCTAVE_SCALE = 1.5
        for n in range(-7,1):
            for i in range(self.iterations):
                # gradients = self._get_gradients(self.img_tensor).datanew_size.astype
                # self.img_tensor.data = self.img_tensor.data + self.lr * gradients.data
                
                new_size = size * (OCTAVE_SCALE**n)
                # img_tensor = img_tensor.resize(new_size.astype(int), Image.ANTIALIAS)
                torchvision.transforms.functional.resize(img_tensor, new_size) # .astype(int)
                # self.img_pil = dream(self.img_pil, self.model, self.layer, 50, 0.05, out_channels = None)

                roll_x = np.random.randint(img_numpy.shape[0])
                roll_y = np.random.randint(img_numpy.shape[1])
                img_roll = np.roll(np.roll(img_tensor.detach().cpu().numpy().transpose(1,2,0), roll_y, 0), roll_x, 1)
                img_roll_tensor = torch.tensor(img_roll.transpose(2,0,1), dtype = torch.float).to(self.device)
                gradients_np = self._get_gradients(img_roll_tensor).detach().cpu().numpy()
                gradients_np = np.roll(np.roll(gradients_np, -roll_y, 1), -roll_x, 2)
                gradients_tensor = torch.tensor(gradients_np).to(self.device)
                img_tensor.data = img_tensor.data + self.lr * gradients_tensor.data

            # make pillow image
            img_out = img_tensor.detach().cpu()
            img_out = img_out.numpy().transpose(1,2,0)
            img_out = np.clip(img_out, 0, 1)
            img_out = Image.fromarray(np.uint8(img_out * 255))
            self.img_pil = img_out

            
        
            

    def plot(self):
        # =============================================================================
        # Plot torch dream
        # =============================================================================
        # img = img.resize(orig_size)
        
        fig = plt.figure(figsize = (size , size))
        plt.imshow(self.img_pil)
