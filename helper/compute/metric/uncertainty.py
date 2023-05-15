import torch

class UncertaintyMetric():
    
    def __init__(self, n_noise=4, n_repeat=2, n_output_neurons=2):
        # works for segmentation, produces uncertainty maps (batch size x 1 x width x height)
        
        self.n_noise = n_noise # amount of times we put an image into the cnn
        self.n_repeat = n_repeat # double the image for some reason
        self.n_output_neurons = n_output_neurons # output of neural network (aka classes)
        
        
    def run(self, model, img_batch):
        
        #print(img_batch.shape)
        
        if len(img_batch.shape) == 4:
            batch_size, _, width, height = img_batch.shape
        elif len(img_batch.shape) == 3:
            batch_size, width, height = img_batch.shape
            
        #print(self.n_output_neurons)
            
        ema_batch = img_batch.repeat(self.n_repeat, 1, 1 ,1)
        
        batch_size_repeat = ema_batch.shape[0]
        
        ema_preds = torch.zeros( (batch_size*self.n_noise*self.n_repeat), self.n_output_neurons, width, height )
        
        for i in range (self.n_noise):
    
            noise = torch.clamp(torch.randn_like(ema_batch) * 0.1, min=-0.2, max=0.2)
            ema_input = (ema_batch + noise)
            
            #print(ema_input.shape)

            with torch.no_grad():
                ema_preds[ (self.n_repeat * batch_size * i) : (self.n_repeat * batch_size * (i+1)) ] = model(ema_input)
        
        ema_preds = torch.nn.functional.softmax(ema_preds, dim=1)
        ema_preds = ema_preds.reshape(self.n_repeat*self.n_noise, batch_size, self.n_output_neurons, width, height)
        ema_preds = torch.mean(ema_preds, dim=0)
        
        entropy = -torch.sum(ema_preds * torch.log(ema_preds), dim=1, keepdim=True)
        
        return entropy # uncertainty map