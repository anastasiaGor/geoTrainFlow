class PyLiDataModule(pl.LightningDataModule):
    def __init__(self, cloud_data_sets, data_geometry, features_to_add_to_sample, auxiliary_features, height, width, batch_size) :
        super().__init__()
        self.cloud_data_sets = cloud_data_sets
        self.data_geometry = data_geometry
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.list_of_xr_datasets = [xr.Dataset() for i in range(len(self.cloud_data_sets))]
        self.list_of_torch_datasets = [{} for i in range(len(self.cloud_data_sets))]
        
    #def prepare_data(self) :
        # preparation of data: mean and std of the dataset (to avoid batch avg), normalization and nan filling
        for i in range(len(self.cloud_data_sets)) :
            # read file
            PERSISTENT_BUCKET = os.environ['PERSISTENT_BUCKET'] 
            if (self.data_geometry =='2D') :
                file_prefix = 'data'
            if (self.data_geometry =='3D') :
                file_prefix = 'data3D_'
            xr_dataset = xr.open_zarr(f'{PERSISTENT_BUCKET}/'+file_prefix+str(i)+'.zarr', chunks='auto')
            rename_rules_dictionary = dict({'votemper':'temp', 'sosstsst':'temp', 'vosaline' : 'saline', 'sosaline' : 'saline'})
            for name_to_replace, new_name in rename_rules_dictionary.items() :
                for var in xr_dataset.variables :
                    if (name_to_replace in var):
                        new_var_name = var.replace(name_to_replace, new_name)
                        xr_dataset = xr_dataset.rename({var : new_var_name})
            xr_dataset = xr_dataset[self.features_to_add_to_sample + self.auxiliary_features + ['mask']]
            for feature in self.features_to_add_to_sample :
                # reapply mask (to avoid issues with nans written in netcdf files)
                xr_dataset[feature] = xr_dataset[feature].where(xr_dataset.mask>0)
                # compute mean, median and std for each level (since temperature/salinity may change a lot with the depth)
                xr_dataset['mean_'+feature] = (xr_dataset[feature].mean(dim=['t', 'x_c', 'y_c']))
                xr_dataset['std_'+feature] = (xr_dataset[feature].std(dim=['t', 'x_c', 'y_c']))
                # fill nans with mean (doesn't the number to be fillted in matter since they will be masked, 
                # but they have to be filled with any numbers so that nans do not propagate everywhere) 
                xr_dataset[feature] = xr_dataset[feature].fillna(xr_dataset['mean_'+feature])
            # save result in a list
            self.list_of_xr_datasets[i] = xr_dataset
            self.list_of_torch_datasets[i] = torchDataset(xr_dataset, self.features_to_add_to_sample, self.auxiliary_features, self.height, self.width)
            
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
        return ([torch.utils.data.DataLoader(dataset, batch_size=len(dataset), drop_last=True, num_workers=0) for dataset in self.test_datasets_byregion] 
                + [torch.utils.data.DataLoader(self.test_dataset_all_data, batch_size=len(self.test_dataset_all_data), num_workers=0)])
    
    def teardown(self, stage : str) :
        if (stage == 'fit') :
            # clean train and val datasets to free memory
            del self.train_dataset, self.val_dataset
        # if (stage == 'test') :
        #     del self.test_datasets   
        # if (stage == 'predict') :
        #     del self.test_datasets   