import torch
from pytorch_lightning.callbacks import Callback
from geoTrainFlow.source.mask_tools import apply_mask_torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import wandb

class LogPredictionsCallback(Callback):
    def __init__(self, logger, feature) :
        super().__init__()
        self.logger = logger
        self.feature = feature
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (batch_idx == 0):
            index_sample = 0
            if ('temp_var_masked' in outputs.keys()) :
                if (len(outputs['temp_var_masked'].shape) == 3) :
                    prediction = outputs['temp_var_masked'][index_sample,:,:].cpu()
                    truth = batch['temp_var'][index_sample,:,:]
                    truth = apply_mask_torch(truth, batch['mask'][index_sample,:,:]).cpu()
                if (len(outputs['temp_var_masked'].shape) == 4) :
                    prediction = outputs['temp_var_masked'][index_sample,0,:,:].cpu()
                    truth = batch['temp_var'][index_sample,0,:,:]
                    truth = apply_mask_torch(truth, batch['mask'][index_sample,0,:,:]).cpu()

            fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(5.,2.5),sharex=True, sharey=True)
            # plot truth
            img = ax[0].imshow(torch.t(truth), cmap='ocean_r', origin='lower')
            fig.colorbar(img, location='left', shrink=0.8)
            ax[0].set(title='Truth')
            color_min = img.colorbar.vmin
            color_max = img.colorbar.vmax
            current_cmap = img.cmap
            current_cmap.set_bad(color='silver')
            # plot prediction
            ax[1].imshow(torch.t(prediction), cmap=current_cmap, \
                         vmin=color_min, vmax=color_max, origin='lower')
            ax[1].set(title='Prediction')
            
            # save plot
            fig.savefig('logs/figure.png')
            self.logger.log_image(key="Example "+str(dataloader_idx), images=[wandb.Image("logs/figure.png")], \
                                  step=None)
            
class LogFirstWeightCallback(Callback) :
    def __init__(self, logger) :
        super().__init__()
        self.logger = logger
    
    def on_train_epoch_end(self, trainer, pl_module):
        first_layer_weights = list(pl_module.torch_model.__dict__['_modules'].values())[0].weight 
        first_weight = np.array(first_layer_weights.cpu().detach().numpy()).flat[0]
        wandb.log(dict({"first_weight" : first_weight}))