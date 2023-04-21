class torchDataset(torch.utils.data.Dataset):
    """Dataset of 2D maps of surface temperature, salinity"""

    def __init__(self, xarray_dataset, features_to_add_to_sample, auxiliary_features, height, width):
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.height = height
        self.width = width
        
        self.data = crop_2d_maps(xarray_dataset, self.height, self.width).load()
        self.data_file_len = len(self.data.t)
        
    def __len__(self):
        return self.data_file_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            list_idx = idx.tolist()
        else :
            list_idx = idx
        selected_time_frames = self.data.isel(t=list_idx) #still xArray object
        
        # create dictionary of a sample (a batch) containig different features in numpy format. 
        # This dictionary is an intermediate step, preparing xArray data for transform into pytorch tensors
        sample = dict()
        sample['mask'] = toTorchTensor(selected_time_frames['mask'].astype(bool))
        sample['eroded_mask'] = toTorchTensor(erode_bin_mask(selected_time_frames['mask']))
        
        for feature in self.features_to_add_to_sample :
            sample['mean_'+feature] = toTorchTensor(self.data['mean_'+feature])
            sample['std_'+feature] = toTorchTensor(self.data['std_'+feature])
            sample[feature] = toTorchTensor(selected_time_frames[feature])

        for feature in self.auxiliary_features :
            sample[feature] = toTorchTensor(selected_time_frames[feature])
    
        return sample
    
    
def toTorchTensor(xrArray):
    transformed_data = torch.tensor(xrArray.values)
    return transformed_data