import torch

class lin_regr_model(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features):
        super().__init__()
        self.data_geometry = data_geometry
        self.nb_of_input_features = nb_of_input_features
        self.nb_of_output_features = nb_of_output_features
        
        self.cut_border_pix_output = None
        self.cut_border_pix_input = None
        
        self.lin1 = torch.nn.Linear(self.nb_of_input_features, self.nb_of_output_features, bias=False)
        
        # initialization 
        self.lin1.weight.data = torch.Tensor([[0.1]])

    def forward(self, x):
        res = pixelize(self.data_geometry, x)
        res = self.lin1(res)
        res = unpixelize(self.data_geometry, res, x.shape)
        return res
    
def pixelize(data_geometry, input_tensor) :
    if (data_geometry == '3D') : #shape [N,L,C,H,W]
        res = input_tensor.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W] where N'=NxL
    if (data_geometry == '2D') :
        res = input_tensor  # shape [N,C,H,W]

    # first split the input 4D torch tensor into individual pixels (equivalent to patches of size 1x1)
    res = torch.nn.functional.unfold(res, kernel_size=1, dilation=1, padding=0, stride=1) 
    res = torch.permute(res, dims=(0,2,1))
    res = torch.flatten(res, end_dim=1).to(torch.float32)
    return res

def unpixelize(data_geometry, input_tensor, dims) :
    if (data_geometry == '3D') :
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = dims
        first_flat_len = batch_len*nb_of_levels
    if (data_geometry == '2D') :
        batch_len, nb_of_channels, output_h, output_w = dims
        first_flat_len = batch_len
    
    # reshape the model output back to a 4D torch tensor
    res = input_tensor.unflatten(dim=0, sizes=[first_flat_len,-1])
    res = torch.permute(res,dims=(0,2,1))
    res = torch.nn.functional.fold(res, output_size=(output_h,output_w), kernel_size=1, dilation=1, padding=0, stride=1)

    if (data_geometry == '3D') :
        # unflatten the levels back
        res = res.unflatten(dim=0, sizes=(dims[0], dims[1]))
    return res