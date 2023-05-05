import numpy as np
import xarray as xr
import torch
import pytorch_lightning as pl
from geoTrainFlow.source.mask_tools import erode_bin_mask

class DataModule(pl.LightningDataModule):
    def __init__(self, path, cloud_data_sets, data_geometry, features_to_add_to_sample, auxiliary_features, \
                 height, width, batch_size, load_data=True) :
        super().__init__()
        self.path = path
        self.cloud_data_sets = cloud_data_sets
        self.data_geometry = data_geometry
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.load_data=load_data
        
        self.list_of_xr_datasets = [xr.Dataset() for i in range(len(self.cloud_data_sets))]
        self.list_of_torch_datasets = [{} for i in range(len(self.cloud_data_sets))]
        
    def prepare_data(self) :
        # preparation of data: mean and std of the dataset (to avoid batch avg), normalization and nan filling
        for i in range(len(self.cloud_data_sets)) :
            # read file
            file_prefix = 'data'
            xr_dataset = xr.open_zarr(self.path+'/'+file_prefix+str(i)+'.zarr', chunks='auto')
            
            # rename variables in xarray 
            rename_rules_dictionary = dict({'votemper':'temp', 'sosstsst':'temp', 'vosaline' : 'saline', 'sosaline' : 'saline'})
            for name_to_replace, new_name in rename_rules_dictionary.items() :
                for var in xr_dataset.variables :
                    if (name_to_replace in var):
                        new_var_name = var.replace(name_to_replace, new_name)
                        xr_dataset = xr_dataset.rename({var : new_var_name})
                        
            # selected only requested variables in xarray
            xr_dataset = xr_dataset[self.features_to_add_to_sample + self.auxiliary_features + ['mask']]
            
            # pre-process features
            for feature in self.features_to_add_to_sample :
                # reapply mask (to avoid issues with nans written in netcdf files)
                xr_dataset[feature] = xr_dataset[feature].where(xr_dataset.mask>0)
                # compute mean, median and std for each level (since temperature/salinity may change a lot with the depth)
                xr_dataset['mean_'+feature] = (xr_dataset[feature].mean(dim=['time_counter', 'x_c', 'y_c']))
                xr_dataset['std_'+feature] = (xr_dataset[feature].std(dim=['time_counter', 'x_c', 'y_c']))
                # fill nans with mean (have to be filled with any numbers so that nans do not propagate everywhere) 
                xr_dataset[feature] = xr_dataset[feature].fillna(xr_dataset['mean_'+feature])
            # save result in a list of xarray datasets
            if self.load_data :
                self.list_of_xr_datasets[i] = xr_dataset.load()
            else :
                self.list_of_xr_datasets[i] = xr_dataset
            # transform xarray datasets into torch datasets
            self.list_of_torch_datasets[i] = iterableTorchDatasetSnapshots(xr_dataset, self.features_to_add_to_sample, self.auxiliary_features, \
                                                                           self.height, self.width)
            
    def setup(self, stage: str) :
        if (stage == 'fit') :
        # takes first 60% of time snapshots for training
            self.train_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(dataset, \
                                                                                     indices=range(0,int(0.6*len(dataset)))) \
                                                                                     for dataset in self.list_of_torch_datasets])
        # takes last 20% of time snapshots for validation (we keep a gap to have validation data decorrelated from trainig data)
            self.val_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(dataset, \
                                                                                     indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                                                     for dataset in self.list_of_torch_datasets])
        # same for test
        if (stage == 'test') :
            self.test_datasets_byregion = [torch.utils.data.Subset(dataset, indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                               for dataset in self.list_of_torch_datasets]
            self.test_dataset_all_data = torch.utils.data.ConcatDataset([torch.utils.data.Subset(dataset, \
                                                                                     indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                                                     for dataset in self.list_of_torch_datasets])
            
    def train_dataloader(self) :
        # create training dataloadder from train_dataset with shuffling with given batch size
        return torch.utils.data.DataLoader(self.train_dataset, \
                                           batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    def val_dataloader(self) :
        # create training dataloadder from val_dataset without shuffling with the same batch size
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, num_workers=0) 
    
    def test_dataloader(self) :
        # create a LIST of dataloaders (a dataloader for each dataset) - to enable diagnostics in each region/season individually 
        # batch size is equal to the dataset length, i.e. there is ONLY 1 batch with all dataset inside (can be better since there is no optimisation in testing)
        return ([torch.utils.data.DataLoader(dataset, batch_size=len(dataset), drop_last=True, num_workers=0) \
                 for dataset in self.test_datasets_byregion] 
                + [torch.utils.data.DataLoader(self.test_dataset_all_data, batch_size=len(self.test_dataset_all_data), num_workers=0)])
    
    def teardown(self, stage : str) :
        if (stage == 'fit') :
            # clean train and val datasets to free memory
            del self.train_dataset, self.val_dataset
        if (stage == 'test') :
            del self.test_datasets_byregion, self.test_dataset_all_data 
            
class iterableTorchDatasetSnapshots(torch.utils.data.Dataset):
    """Iterable torch dataset of temprotal snapshots of selected features from xarray data"""

    def __init__(self, xarray_dataset, features_to_add_to_sample, auxiliary_features, height, width):
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.height = height
        self.width = width
        
        self.data = xarray_dataset.isel(x_c=slice(None,self.width), y_c=slice(None,self.height))
        self.data_file_len = len(self.data.time_counter)
        
    def __len__(self):
        return self.data_file_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            list_idx = idx.tolist()
        else :
            list_idx = idx
        selected_time_frames = self.data.isel(time_counter=list_idx) #still xArray object
        
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