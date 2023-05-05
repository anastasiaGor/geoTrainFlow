import torch

class CNN(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features, padding='same', padding_mode='replicate', \
                 kernel_size=3, int_layer_width=64, activation_function = torch.nn.functional.relu):
        super().__init__()
        self.data_geometry = data_geometry
        self.padding = padding
        self.kernel_size = kernel_size
        self.padding_mode = 'replicate'
        self.activation_function = activation_function
        
        self.cut_border_pix_input = None
        if self.padding == 'same' :
            self.cut_border_pix_output = self.cut_border_pix_input
        if self.padding == 'valid' :
            self.cut_border_pix_output = (self.cut_border_pix_input or 0) + self.kernel_size//2
        
        self.conv1 = torch.nn.Conv2d(in_channels=nb_of_input_features, out_channels=int_layer_width, kernel_size=self.kernel_size, \
                                     padding=self.padding,  padding_mode=self.padding_mode) 
        self.conv2 = torch.nn.Conv2d(int_layer_width, int_layer_width, kernel_size=self.kernel_size, padding='same', padding_mode=self.padding_mode) 
        self.conv3 = torch.nn.Conv2d(int_layer_width, nb_of_output_features, kernel_size=self.kernel_size, padding='same', \
                                     padding_mode=self.padding_mode)

    def forward(self, x):
        batch_len = x.shape[0]
        if (self.data_geometry == '3D') :
            nb_of_levels = x.shape[1]
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
        else :
            res = x
        
        res = self.conv1(res)
        res = self.activation_function(res)
        res = self.conv2(res)
        res = self.activation_function(res)
        res = self.conv3(res)
        
        if (self.data_geometry == '3D') :
            # unflatten the levels
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res       