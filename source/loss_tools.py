import torch
from geoTrainFlow.source.mask_tools import masked_mean

def evaluate_loss_with_mask(data_geometry, metrics, mask, truth, model_output, reduction='mean', normalization=True) :
    if normalization :
        normalization_coef = masked_mean(data_geometry, truth, mask, reduction='normalization_mean')
    else :
        normalization_coef = 1.
    non_reduced_non_masked_metrics = metrics(model_output/normalization_coef, truth/normalization_coef, reduction='none')
    reduced_metrics = masked_mean(data_geometry, non_reduced_non_masked_metrics, mask, reduction=reduction)
    return reduced_metrics

def pressure_based_MSEloss(batch, pred_sigma, target_sigma, cut_border_pix, idx_level=100, normalization=True) :
    if (self.data_geometry != '3D') :
        print('ERROR: pressure based loss is available only for 3D data')
    return

    rho_ct_ct = cut_bords(batch['rho_ct_ct'], cut_border_pix)
    dx = cut_bords(batch['e1t'], cut_border_pix)
    dy = cut_bords(batch['e2t'], cut_border_pix)
    z_l = batch['z_l']
    mask = cut_bords(batch['eroded_mask'], cut_border_pix)

    narrowed_mask = mask[:,1:-1,1:-1] # use narrowed mask since borders are cropped when computing gradient
    
    true_pres_grad_x, true_pres_grad_y, true_pres_grad_norm = get_pressure_grad(target_sigma, rho_ct_ct, dx, dy, z_l)
    pred_pres_grad_x, pred_pres_grad_y, pred_pres_grad_norm = get_pressure_grad(pred_sigma, rho_ct_ct, dx, dy, z_l)

    pres_grad_x_loss = evaluate_loss_with_mask('3D', torch.nn.functional.mse_loss, narrowed_mask, \
                                               pred_pres_grad_x[:,idx_level,:,:], true_pres_grad_x[:,idx_level,:,:], \
                                               reduction='mean', normalization=normalization)
    pres_grad_y_loss = evaluate_loss_with_mask('3D', torch.nn.functional.mse_loss, narrowed_mask, \
                                               pred_pres_grad_y[:,idx_level,:,:], true_pres_grad_y[:,idx_level,:,:], \
                                               reduction='mean', normalization=normalization)
    loss_pres_grad = pres_grad_x_loss + pres_grad_y_loss
    return loss_pres_grad


def mixed_pressure_loss(moduleObject, batch, y_model, y_true, idx_level=100, normalization=True, alpha=1.) :
    index_of_temp_var_feature = moduleObject.output_features.index('temp_var')
    
    pred_sigma = y_model[:, :, index_of_temp_var_feature, :, :]
    target_sigma = y_true[:, :, index_of_temp_var_feature, :, :]   
    
    loss_pres_grad = pressure_based_MSEloss(batch, pred_sigma, target_sigma, moduleObject.torch_model.cut_border_pix_output, \
                                            idx_level=idx_level, normalization=normalization)
    
    loss_value = evaluate_loss_with_mask('3D', torch.nn.functional.mse_loss, mask, y_model, y_true, \
                           reduction='mean', normalization=normalization)
    
    # get the mixed loss that will be optimized in training
    # alpha is a coefficient that weights the contribution of pressure grad error
    loss_mixed = alpha*loss_pres_grad+loss_value
    
    loss_dictionary = dict({'loss_train' : loss_mixed,
                 'loss_pressure' : loss_pres_grad,
                 'loss_value' : loss_value})
    
    return loss_dictionary