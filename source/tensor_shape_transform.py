import torch

def cut_bords(tensor, nb_of_border_pix) :
    if nb_of_border_pix is None :
        return tensor
    else :
        if (len(tensor.shape) == 5) :
            return tensor[:,:, :, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        if (len(tensor.shape) == 4) :
            return tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        if (len(tensor.shape) == 3) :
            return tensor[:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        
def expand_to_bords(tensor, nb_of_border_pix) :
    if nb_of_border_pix is None :
        return tensor
    else :
        if (len(tensor.shape) == 4) :
            new_tensor = torch.empty((tensor.shape[0],tensor.shape[1], tensor.shape[2]+2*nb_of_border_pix, tensor.shape[3]+2*nb_of_border_pix)).\
            to(tensor.device)
            new_tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] = tensor
        if (len(tensor.shape) == 3) :
            new_tensor = torch.empty((tensor.shape[0], tensor.shape[1]+2*nb_of_border_pix, tensor.shape[2]+2*nb_of_border_pix)).to(tensor.device)
            new_tensor[:,nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] = tensor        
        return new_tensor
    
def transform_and_stack_features(batch, features, nb_of_border_pix, normalization_features=None) :
    # check if normalization is needed
    for index, feature in enumerate(features) :
        if feature.startswith('normalized_') :
            not_normalized_feature_name = feature.replace("normalized_", "")
            if (normalization_features is None) :
                norm_feature = None
            else :
                norm_feature = normalization_features[index]
                if not(norm_feature in batch.keys()) :
                    batch = add_transformed_feature(batch, norm_feature)
            batch['normalized_'+not_normalized_feature_name] = tensor_normalize(batch[not_normalized_feature_name], batch, \
                                                                                not_normalized_feature_name, norm_feature)
        if feature.startswith('filtered_') :
            not_filt_feature_name = feature.replace("filtered_", "")
            batch['filtered_'+not_filt_feature_name] = filter_with_convolution(batch[not_filt_feature_name], convolution_kernel_size=3)
    # stack features from sample into channel (create channel dimension in tensor)
    stacked_channels = torch.stack([cut_bords(batch[key], nb_of_border_pix) for key in features])
    if (len(stacked_channels.shape) == 4): # 2d data case -> 4D cubes 
        transform = torch.permute(stacked_channels, (1,0,2,3)).to(torch.float32) #shape [N,C,H,W]
    if (len(stacked_channels.shape) == 5): # 3d data case -> 5d cubes
        transform = torch.permute(stacked_channels, (1,2,0,3,4)).to(torch.float32) #shape [N,L,C,H,W]
    return transform
