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
        if (self.data_geometry == '3D') :
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = x.shape
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
            new_batch_len = batch_len*nb_of_levels
        if (self.data_geometry == '2D') :
            new_batch_len, nb_of_channels, output_h, output_w = x.shape
            res = x 
        
        # first split the input 4D torch tensor into individual pixels (equivalent to patches of size 1x1)
        res = torch.nn.functional.unfold(res, kernel_size=1, dilation=1, padding=0, stride=1)
        res = torch.permute(res, dims=(0,2,1))
        res = torch.flatten(res, end_dim=1).to(torch.float32)
        
        # perform linear regression
        res = self.lin1(res)
        
        # reshape the model output back to a 4D torch tensor
        res = torch.permute(res.unflatten(dim=0, sizes=[new_batch_len,-1]),dims=(0,2,1))
        res = torch.nn.functional.fold(res, output_size=(output_h,output_w), kernel_size=1, dilation=1, padding=0, stride=1)
        
        if (self.data_geometry == '3D') :
            # unflatten the levels back
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res