import torch
import pytorch_lightning as pl
from geoTrainFlow.source.tensor_shape_transform import cut_bords, expand_to_bords, transform_and_stack_features
from geoTrainFlow.source.loss_tools import evaluate_loss_with_mask, pressure_based_MSEloss
from geoTrainFlow.source.mask_tools import apply_mask_torch
from geoTrainFlow.source.custom_tensor_operators import finite_diffs_sqr_2d_map

class TrainingModule(pl.LightningModule):
    def __init__(self, torch_model, input_features, output_features, output_units, loss, optimizer, learning_rate, \
                 input_normalization_features=None, loss_normalization=False):
        super().__init__()
        self.torch_model = torch_model
        self.input_features = input_features
        self.output_features = output_features
        self.output_units = output_units
        self.input_normalization_features = input_normalization_features
        self.loss_normalization = loss_normalization
        
        self.loss = loss
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_normalization = loss_normalization
        
        ## initialization of weights
        #torch_model.weight.data = torch.Tensor([1.0])

        #construct list of names of features to be predicted
        self.list_of_features_to_predict=list()
        for i, feature in enumerate(self.output_features) :
            self.list_of_features_to_predict.append(feature)
            if feature.startswith('normalized_') :
                # if model output is a normalized feature then compute also the non-normalized feature (for the diagnostics)
                not_normalized_feature = feature.replace("normalized_", "")
                self.list_of_features_to_predict.append(not_normalized_feature)
                
        self.data_geometry = self.torch_model.data_geometry
    
    def common_step(self, batch, batch_idx) :
        x = transform_and_stack_features(batch, self.input_features, self.torch_model.cut_border_pix_input, self.input_normalization_features)
        y_true = transform_and_stack_features(batch, self.output_features, self.torch_model.cut_border_pix_output)
        mask = cut_bords(batch['eroded_mask'], self.torch_model.cut_border_pix_output)

        if (self.output_units is None) :
            y_model = self.torch_model(x)
        else :
            y_units = transform_and_stack_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            y_model = y_units*self.torch_model(x)
        
        logs = dict()
        if (self.loss=='pressure_based_MSEloss') :
            loss_dict = mixed_pressure_loss(self, batch, y_model, y_true, idx_level=100, normalization=True, alpha=1.)
            logs = logs | loss_dict
        else :
            loss_value = evaluate_loss_with_mask(self.data_geometry, self.loss, mask, y_model, y_true, \
                                               reduction='mean', normalization=self.loss_normalization)  
            logs = logs | dict({'loss_train' : loss_value})
        return logs
        
    def training_step(self, batch, batch_idx) :
        logs = self.common_step(batch, batch_idx)
        self.log_dict(logs, on_step=False, on_epoch=True)
        return logs['loss_train']

    # validation logics (is evaluated during the training, but the data is not used to the optimization loop)
    def validation_step(self, batch, batch_idx) :
        logs = self.common_step(batch, batch_idx)
        self.log_dict(dict({'loss_validation' : logs['loss_train']}), on_step=False, on_epoch=True)
    
    # gives model output in a form of a dictionary of batches of 2d fields
    def predict_step(self, batch, batch_idx, dataloader_idx) :
        x = transform_and_stack_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        
        output_tensor = self.torch_model(x)
        if (self.data_geometry == '2D') :
            batch_len, nb_of_channels, output_h, output_w = output_tensor.shape
        if (self.data_geometry == '3D') :
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = output_tensor.shape

        if not(self.output_units is None) : # if output of the model is dimensionless -> compute output with physical units
            y_units = transform_and_stack_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            output_tensor_units = output_tensor*y_units
            
        # construct the dictionary of the predicted features by decomposing the channels into dictionary entities
        pred = dict()
        if (self.data_geometry == '2D') :
            channel_dim = 1
        if (self.data_geometry == '3D') :
            channel_dim = 2
        for i, feature in enumerate(self.output_features) :
            if (self.output_units is None) :
                pred[feature] = output_tensor.select(dim=channel_dim, index=i)
            else :
                # save dimensionless result
                pred[feature+'_dimless'] = output_tensor.select(dim=channel_dim, index=i)
                # save result with physocal units
                pred[feature] = output_tensor_units.select(dim=channel_dim, index=i)
            # if some outputs are normalized then compute also result in the restored units (not normalized)
            if feature.startswith('normalized_') :
                not_normalized_feature_name = feature.replace("normalized_", "")
                pred[not_normalized_feature_name] = PyLiDataModule.tensor_restore_norm(pred[feature], batch, not_normalized_feature)
        
        # save the mask and masked outputs (use the eroded mask)
        for i, feature in enumerate(self.list_of_features_to_predict) :
            if (self.data_geometry == '2D') :
                pred['eroded_mask'] = batch['eroded_mask']
                pred['mask'] = batch['mask']
            if (self.data_geometry == '3D') :
                pred['eroded_mask'] = batch['eroded_mask'][:,None,:,:]
                pred['mask'] = batch['mask'][:,None,:,:]
            pred[feature+'_masked'] = expand_to_bords(pred[feature], self.torch_model.cut_border_pix_output)
            pred[feature+'_masked'] = apply_mask_torch(pred[feature+'_masked'], pred['eroded_mask'])
        return pred 
    
    # testing logic - to evaluate the model after training
    def test_step(self, batch, batch_idx, dataloader_idx) :
        pred = self.predict_step(batch, batch_idx, dataloader_idx)
        mask = cut_bords(batch['eroded_mask'], self.torch_model.cut_border_pix_output)
        
        test_dict = dict({'loss_val' : dict(), 'loss_grad' : dict(), 'corr_coef' : dict(), 'corr_coef_grad' : dict()})
        dict_for_log = dict()
        
        # global metrics
        for i, feature in enumerate(self.list_of_features_to_predict) :
            truth = cut_bords(batch[feature], self.torch_model.cut_border_pix_output)
            model_output = pred[feature]  # use unmasked prediction here, mask will be applied further on error tensor
            
            # compute standard MSE
            test_dict['loss_val'][feature] = evaluate_loss_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                                    mask, model_output, truth, \
                                                                                    reduction='mean', normalization=True)
            # compute correlation coefficient
            test_dict['corr_coef'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output).view(1,-1), \
                                                              torch.flatten(truth).view(1,-1))))[1,0]
            # metrics on horizontal gradients
            model_output_grad = finite_diffs_sqr_2d_map(model_output)
            truth_grad = finite_diffs_sqr_2d_map(truth)
            test_dict['loss_grad'][feature] = evaluate_loss_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                     mask[:,1:-1,1:-1], model_output_grad, truth_grad, \
                                                                      reduction='mean', normalization=True)

        # for 3d data - a specific test: MSE of pressure gradient at 100th level
        if (self.data_geometry == '3D') :
            idx_level = 100
            true_temp_var = cut_bords(batch['temp_var'], self.torch_model.cut_border_pix_output)
            model_temp_var = pred['temp_var']
            test_dict['loss_val']['pressure_grad'] = pressure_based_MSEloss(batch, model_temp_var, true_temp_var, \
                                                    self.torch_model.cut_border_pix_output, \
                                                    idx_level=100, normalization=True)
        for metrics in list(test_dict.keys()) : 
            for feature in list(test_dict[metrics].keys()) : 
                dict_for_log.update({(metrics+'_'+feature) : test_dict[metrics][feature]})
        self.log_dict(dict_for_log)

    def configure_optimizers(self) :
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer