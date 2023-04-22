# https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py

import torch
import torch.nn # import ReLU
from torch.autograd import Variable
from torchvision import models


import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im








def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def apply_heatmap(R, sx, sy):
    """
        Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial
        This is (so far) only used for LRP
    """
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    return heatmap
    # plt.show()


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr




    


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency





###################################

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
       #####
    """
    def __init__(self, model, experiment_name="tmp"):
        
        self.experiment_name = experiment_name
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()
        
        self.gradients_as_arr = None

    def hook_layers(self):
        ####
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        print("*"*50)
        # print(self.model.decent_blocks[0][0]) # ccc

        first_conv_layer = self.model.decent_block.decent_block_116[0][0] # self.model.decent_blocks[1][0][0] # self.model.decent_block.decent_block_116[1]

        first_layer = first_conv_layer #  list(self.model.features._modules.items())[0][1]
        
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        #####
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        ####### xxx aaa
        print(self.model.decent_block.decent_block_reduction._modules.items())
        # print(dict(self.model.decent_blocks[0].named_modules()))

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.decent_block.decent_block_reduction._modules.items():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def run(self, input_image, target_class, cnn_layer, filter_pos, layer):

        print("cnn_layer", cnn_layer)
        print("target_class", target_class)
        #####
        self.model.zero_grad()
        # Forward pass
        x = input_image
        # decent_block.decent_block_116.0.0
        for index, layer in enumerate(layer): # self.model.features
            print("*"*50)
            print("layer", layer)
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        
        self.grad_image = self.gradients.data.numpy()[0]
        
        self.grad_image_gray = convert_to_grayscale(self.grad_image)
        
        self.grad_image_pos, self.grad_image_neg = get_positive_negative_saliency(self.grad_image)
    
    def save(self, results_path):
        
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
        self.experiment_name
        
        def save_gradient_images(gradient, path):
            # Normalize
            gradient = gradient - gradient.min()
            gradient /= gradient.max()
            # Save image
            if isinstance(im, (np.ndarray, np.generic)):
                im = format_np_output(im)
                im = Image.fromarray(im)
            im.save(path)
        
        # =============================================================================
        # Colored and grayscale gradients
        # =============================================================================
        
        # save normal gradient image
        p =  os.path.join(results_path, file_name_to_export + '_Guided_BP_color.png')
        save_gradient_images(guided_grads, p)
        
        # Convert to grayscale
        grayscale_guided_grads = convert_to_grayscale(guided_grads)
        save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')

        # =============================================================================
        # Saliency Maps
        # =============================================================================
        # Positive and negative saliency maps
        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
        save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')



        # =============================================================================
        # Grad CAM
        # =============================================================================
        #print(original_image.size)
        #print(grayscale_guided_grads.squeeze().shape)
        # save_class_activation_images(pil_im, grayscale_guided_grads.squeeze(), file_name=file_name_to_export + '_grad')
        # def save_class_activation_images(org_img, activation_map, file_name):
        """
            Saves cam activation map and activation map on the original image
        Args:
            org_img (PIL img): Original image
            activation_map (numpy arr): Activation map (grayscale) 0-255
            file_name (str): File name of the exported image
        """
        org_img = original_image.resize((224, 224), Image.ANTIALIAS)


        # Grayscale activation map
        heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
        # Save colored heatmap
        path_to_file = os.path.join(results_path, file_name+'_grad_Cam_Heatmap.png')
        save_image(heatmap, path_to_file)
        # Save heatmap on iamge
        path_to_file = os.path.join(results_path, file_name+'_grad_Cam_On_Image.png')
        save_image(heatmap_on_image, path_to_file)
        # Save grayscale heatmap
        path_to_file = os.path.join(results_path, file_name+'_grad_Cam_Grayscale.png')
        save_image(activation_map, path_to_file)

