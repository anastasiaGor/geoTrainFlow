def erode_bin_mask(xr_mask) :
    erosion_structure_matrix = np.array([(0,0,1,0,0), (0,1,1,1,0), (1,1,1,1,1), (0,1,1,1,0), (0,0,1,0,0)])
    np_array_mask = ndimage.binary_erosion(xr_mask, structure=erosion_structure_matrix)
    return xr_mask.copy(data=np_array_mask)

def masked_mean(data_geometry, tensor, mask, reduction='mean') :
    if (data_geometry == '2D') :
        if (len(tensor.shape) == 4) : # 4D tensor with C features
            batch_len, nb_of_channels, output_h, output_w = tensor.shape  
            channel_dim = 1
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_channels
            mask = mask[:,None,:,:]
        if (len(tensor.shape) == 3) : # 1 feature (=1 channel)-> 3D tensor
            batch_len, output_h, output_w = tensor.shape  
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask
    if (data_geometry == '3D') :
        if (len(tensor.shape) == 5) : # full 5D tensor
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = tensor.shape  
            channel_dim = 2
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels*nb_of_channels
            mask = mask[:,None,None,:,:]
        if (len(tensor.shape) == 4) : # 1 feature (=1 channel)
            batch_len, nb_of_levels, output_h, output_w = tensor.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels
            mask = mask[:,None,:,:]
        if (len(tensor.shape) == 3) : # 1 feature (=1 channel) and 1 level
            batch_len, output_h, output_w = tensor.shape  
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask

    if (reduction=='none') :
        return (tensor*mask)
    
    total = torch.sum(tensor*mask)
    if (reduction=='sum') : 
        return total    # 1 number
    if (reduction=='mean') : 
        return (total/valid_mask_counts) # 1 number
    if (reduction=='vertical_mean') :
        sum_over_each_layer = torch.sum(tensor*mask, dim=(-2,-1))
        valid_counts_each_layer = torch.count_nonzero(mask, dim=(-2,-1))
        vertical_profile_of_each_sample = sum_over_each_layer/valid_counts_each_layer # shape [N,L,(C)]
        res = torch.mean(vertical_profile_of_each_sample, dim=0) # average over the batch, final shape [L, (C)]
        return res
    if (reduction=='horizontal_mean') :
        sum_over_depth_at_each_point = torch.sum(tensor*mask, dim=1) # avg over depth, shape [N, (C), H, W]
        valid_counts = torch.count_nonzero(mask, dim=1)*nb_of_levels
        horizontal_error_of_each_sample = sum_over_depth_at_each_point/valid_counts
        res = torch.mean(horizontal_error_of_each_sample, dim=0) # [(C), H, W]
        return res 
    if (reduction=='sample_mean') :
        reduc_dims = tuple(range(1,len(tensor.shape))) # all dimensions>0
        if not (channel_dim is None) :
            reduc_dims.remove(channel_dim) 
        sum_over_each_sample = torch.sum(tensor*mask, dim=reduc_dims)
        valid_counts = torch.count_nonzero(mask, dim=reduc_dims)*(nb_of_levels if data_geometry == '3D' else 1.)
        res = sum_over_each_sample/valid_counts # shape[N, (C)]
        return res
    if (reduction=='channel_mean') : 
        reduc_dims = list(range(len(tensor.shape)))
        reduc_dims.remove(channel_dim)
        sum_over_each_channel = torch.sum(tensor*mask, dim=reduc_dims)  
        valid_counts = torch.count_nonzero(mask, dim=reduc_dims)*(nb_of_levels if data_geometry == '3D' else 1.)
        res = sum_over_each_channel/valid_counts # shape [C]
        return res
    if (reduction=='normalization_mean') : # by channel, individual for sample (and level if 3D)
        sum_over_each_sample = torch.sum(tensor*mask, dim=(-2,-1)) #shape [N, (L), (C)]
        valid_counts = torch.count_nonzero(mask, dim=(-2,-1))
        mean_of_sample = sum_over_each_sample/valid_counts
        res = torch.unsqueeze(torch.unsqueeze(mean_of_sample, dim=-1), dim=-1) # shape [N, (L), (C), 1, 1]
        return res 