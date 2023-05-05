import torch

def central_diffs(dataArray) :
    if len(dataArray.shape) == 5 : #5d data cube
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = dataArray.shape
        flatten_data = dataArray.flatten(start_dim=0, end_dim=2)[:,None,:,:]
    if len(dataArray.shape) == 4 : # 1 channel (or 1 level)
        batch_len, nb_of_channels, width, height = dataArray.shape
        flatten_data = dataArray.flatten(start_dim=0, end_dim=1)[:,None,:,:]
    if len(dataArray.shape) == 3 : # 1 channel
        batch_len, width, height = dataArray.shape
        flatten_data = dataArray[:,None,:,:]
    weights = torch.zeros(2,1,3,3).to(dataArray.device) # 2 channels : 1 channel for x-difference, other for y-differences
    weights[0,0,:,:] = torch.tensor([[0,0.,0],[-0.5,0.,0.5],[0,0.,0]]) #dx
    weights[1,0,:,:] = torch.tensor([[0,-0.5,0],[0,0.,0],[0,0.5,0]])   #dy
    res = torch.nn.functional.conv2d(flatten_data.float(), weights, \
                               bias=None, stride=1, padding='same', dilation=1, groups=1)
    if len(dataArray.shape) == 5 :
        res_dx = res[:,0,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_levels, nb_of_channels))
        res_dy = res[:,1,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_levels, nb_of_channels))
    if len(dataArray.shape) == 4 :
        res_dx = res[:,0,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_channels))
        res_dy = res[:,1,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_channels))
    if len(dataArray.shape) == 3 :
        res_dx = res[:,0,1:-1,1:-1]
        res_dy = res[:,1,1:-1,1:-1]
    return res_dx, res_dy

def finite_diffs_sqr_2d_map(dataArray) :
    res_dx, res_dy = central_diffs(dataArray)
    res = torch.pow(res_dx,2) + torch.pow(res_dy,2)
    return res

def get_pressure_grad(temp_var, rho_ct_ct, dx, dy, z_l) :
    g = 9.81
    dz = torch.diff(z_l)
    delta_rho = 0.5*temp_var*rho_ct_ct
    dx_rho, dy_rho = central_diffs(delta_rho)
    dx_rho = dx_rho[:,:-1,:,:]/dx[:,None,1:-1,1:-1]
    dy_rho = dy_rho[:,:-1,:,:]/dy[:,None,1:-1,1:-1]
    dx_p = torch.cumsum(dx_rho*g*dz[:,:,None,None], axis=1)   
    dy_p = torch.cumsum(dy_rho*g*dz[:,:,None,None], axis=1)
    return [dx_p, dy_p, torch.sqrt(torch.pow(dx_p,2)+torch.pow(dy_p,2))]

def filter_with_convolution(tensor, convolution_kernel_size=3) :
    if (len(tensor.shape) == 4) : #[N,L,H,W]
        batch_len, nb_levels, height, width = tensor.shape
        flatten_tensor = tensor.flatten(start_dim=0, end_dim=1)[:,None,:,:]
    if (len(tensor.shape) == 3) : #[N,H,W]
        flatten_tensor = tensor[:,None,:,:]
        
    weights = torch.ones(1,1,convolution_kernel_size,convolution_kernel_size).to(tensor.device) #matrix filled with ones for averaging
    padding = torch.nn.ReplicationPad2d(convolution_kernel_size//2)  #pad borders with 1 row/column with the replicated values
    padded_tensor= padding(flatten_tensor)
    res = torch.nn.functional.conv2d(padded_tensor, weights, bias=None, stride=1, padding='valid', dilation=1, groups=1)
    res = res[:,0,:,:]
    if (len(tensor.shape) == 4) :
        res = res.unflatten(dim=0, sizes=(batch_len, nb_levels))
    return res
    
def add_transformed_feature(batch, missing_feature_name) :
    if missing_feature_name.startswith('filtered_') :
        not_filt_feature_name = missing_feature_name.replace("filtered_", "")
        batch['filtered_'+not_filt_feature_name] = filter_with_convolution(batch[not_filt_feature_name], convolution_kernel_size=3)
    if missing_feature_name.startswith('sqrt_filtered_') :
        feature_name = missing_feature_name.replace("sqrt_filtered_", "")
        batch['sqrt_filtered_'+feature_name] = torch.sqrt(filter_with_convolution(batch[feature_name], convolution_kernel_size=3))
    return batch