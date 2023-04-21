class FCNN(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features, input_patch_size, output_patch_size, \
                 activation_function = torch.nn.functional.relu, int_layer_width=50):
        super().__init__()
        self.data_geometry = data_geometry
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.activation_function = activation_function
        self.int_layer_width = int_layer_width
        
        self.lin1 = torch.nn.Linear(nb_of_input_features*input_patch_size**2, int_layer_width, bias=True)
        self.lin2 = torch.nn.Linear(int_layer_width, int_layer_width, bias=True)
        self.lin3 = torch.nn.Linear(int_layer_width, nb_of_output_features*output_patch_size**2, bias=True)
        
        self.cut_border_pix_output = self.input_patch_size//2 - self.output_patch_size//2
        if (self.cut_border_pix_output < 1) :
            self.cut_border_pix_output = None
        self.cut_border_pix_input = None

    def forward(self, x):
        if (self.data_geometry =='3D') :
            batch_len, nb_of_levels, nb_of_channels = x.shape[0:3]
            output_h = x.shape[3]-2*(self.cut_border_pix_output or 0)
            output_w = x.shape[4]-2*(self.cut_border_pix_output or 0)
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
            new_batch_len = batch_len*nb_of_levels
        if (self.data_geometry =='2D') :
            new_batch_len, nb_of_channels = x.shape[0:2]
            output_h = x.shape[2]-2*(self.cut_border_pix_output or 0)
            output_w = x.shape[3]-2*(self.cut_border_pix_output or 0)
            res = x
        
        # create patches of size 'input_patch_size' and join them into batches (zero padding - will remove border pixels)
        res = torch.nn.functional.unfold(res, kernel_size=self.input_patch_size, dilation=1, padding=0, stride=1)
        res = torch.permute(res, dims=(0,2,1))
        res = torch.flatten(res, end_dim=1)
        
        # pass though the FCNN
        res = self.lin1(res)
        res = self.activation_function(res)
        res = self.lin2(res)
        res = self.activation_function(res)
        res = self.lin3(res)
        
        # reshape the output patches back into a 4D torch tensor
        res = res.unflatten(dim=0, sizes=(new_batch_len,-1))
        res = torch.permute(res,dims=(0,2,1))
        res = torch.nn.functional.fold(res, output_size=(output_h,output_w), \
                                       kernel_size=self.output_patch_size, dilation=1, padding=0, stride=1)
        # compute the divider needed to get correct values in case of overlapping patches (will give mean over all overlapping patches)
        mask_ones = torch.ones((1,1,output_h,output_w)).to(x.device)
        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(mask_ones, kernel_size=self.output_patch_size), \
                                           kernel_size=self.output_patch_size, output_size=(output_h,output_w))   
        res = res/divisor.view(1,1,output_h,output_w)
        
        if (self.data_geometry =='3D') :
            # unflatten the levels
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        
        return res