import torch

def tensor_restore_norm(tensor, batch, reference_feature, normalization_feature=None) :
    if (len(tensor.shape) == 3) :
        std = batch['std_'+reference_feature][:,None,None]
        mean = batch['mean_'+reference_feature][:,None,None]
    if (len(tensor.shape) == 4) :
        std = batch['std_'+reference_feature][:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,None,None]
    if (len(tensor.shape) == 5) :
        std = batch['std_'+reference_feature][:,:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,:, None,None]
    if (normalization_feature is None) :
        return tensor*std+mean
    else : 
        return tensor*batch[normalization_feature]+mean

def tensor_normalize(tensor, batch, reference_feature, normalization_feature=None) :
    if (len(tensor.shape) == 3) :
        std = batch['std_'+reference_feature][:,None,None]
        mean = batch['mean_'+reference_feature][:,None,None]
    if (len(tensor.shape) == 4) :
        std = batch['std_'+reference_feature][:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,None,None]
    if (len(tensor.shape) == 5) :
        std = batch['std_'+reference_feature][:,:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,:, None,None]
    if (normalization_feature is None) :
        return (tensor-mean)/std
    else : 
        return (tensor-mean)/batch[normalization_feature]