import tyc_2DGCN_Model
import tyc_3DGCN_Model
from deepchem.models import optimizers


graph_conv_layers=[64,30]
mode="regression"
n_tasks=1
bs=5470
pred_h_feats=32
log_frequency=5
learning_rate=0.001
optimizer=optimizers.Adam(learning_rate=learning_rate)

model_2d=tyc_2DGCN_Model.tyc_2DGCN_Model(mode=mode,n_tasks=n_tasks,batch_size=bs,predictor_hidden_feats=pred_h_feats,
                                         log_frequency=log_frequency,optimizer=optimizer,graph_conv_layers=graph_conv_layers,batchnorm=True)

total_parameters_1 = sum(p.numel() for p in model_2d.model.parameters() if p.requires_grad)
print(f"Total Parameters model 1: {total_parameters_1}")