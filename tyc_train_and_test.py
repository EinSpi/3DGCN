import deepchem.molnet as dcmolnet
from datetime import datetime
from deepchem.models import optimizers
from deepchem.models.callbacks import ValidationCallback
from deepchem.metrics import Metric
from deepchem.metrics import pearson_r2_score
from deepchem.metrics import mean_absolute_error
from tyc_easy_3DGCN_Model import tyc_easy_3DGCN_Model
from tyc_3DGCN_Model import tyc_3DGCN_Model
from tyc_root_featurizer import tyc_MolGraphConvFeaturizer
from tyc_2DGCN_Model import tyc_2DGCN_Model
import deepchem as dc
def l2_norm(self):
    lambda_reg=0.00001
    l2_penalty = sum(p.pow(2.0).sum() for p in self.model.parameters())
    return lambda_reg * l2_penalty

#DATASET PREPARE
tasks,dataset,transformers=dcmolnet.load_qm7(featurizer=tyc_MolGraphConvFeaturizer(),reload=False)
train,valid,test=dataset
#HYPERPARAMS
num_epoch=200# epochs
bs=64#batch size
log_frequency=2# how many steps to log: required step number for single epoch= size of dataset/batch size
mode='regression'# using qm7, this is regression
n_tasks=len(tasks)#how many tasks
pred_h_feats=64#how many hidden feats are there in predictor part of each model
scheduler=optimizers.ExponentialDecay(initial_rate=0.001,decay_steps=100,decay_rate=0.99)#scheduler
scheduler=0.001#if you don't want to use lr scheduler,set this to a simple float, if you want to use scheduler above, commentize this line.
optimizer=optimizers.Adam(learning_rate=scheduler)
graph_conv_layers=[64,30] #must be a integerlist indicating the hidden feature numbers of each gcn_layer, must end with 30, can be arbitrary long, if None, it returns [30], 3dgcn model doesn't have this feature.
metric = Metric(pearson_r2_score)#Metric
patienc=5#patience to early quit
i_changed_hyper_params_above=True#!!!!!!!!!!!!IF YOU CHANGED ABOVE PARAS, SET THIS TO TRUE TO LOG THE HP CHANGES IN LOSS_LIST FILE!!!!!!!!


#BASIC SETTINGS, NOT CONSIDER AS HYPERPARAMS
model_dir_1 = "E:\\WS2324\\seminar\\Experiments\\2D_GCN\\num_gcn=2\\gcn_hidden_size=%d\\pred_h_feats=%d" % (graph_conv_layers[0], pred_h_feats)
#model_dir_1="E:\WS2324\seminar\Tyc_2DGCN"#directory to save model 1
#model_dir_2="E:\WS2324\seminar\Tyc_easy_3DGCN"#...................2
#model_dir_3="E:\\WS2324\\seminar\\Experiments\\3D_GCN\\num_gcn=1\\gcn_hidden_size=%d" % pred_h_feats#........................3
checkpoint_interval=10
max_checkpoints_to_keep=200
callback=[]#Callback functions, if you want to note lr to tensorboard for every epoch, please set this to [log_lr_to_db]
all_losses_1=[]# temporary loss container for model 1
all_losses_2=[]# ...................................2
all_losses_3=[]# ...................................3


#PREPARE MODELS
print("construct models")
model_1=tyc_2DGCN_Model(mode=mode,n_tasks=n_tasks,batch_size=bs,predictor_hidden_feats=pred_h_feats,log_frequency=log_frequency,optimizer=optimizer,model_dir=model_dir_1,tensorboard=True,graph_conv_layers=graph_conv_layers,batchnorm=True,regularization_loss=l2_norm)
#model_2=tyc_easy_3DGCN_Model(mode=mode,n_tasks=n_tasks,batch_size=bs,predictor_hidden_feats=pred_h_feats,log_frequency=log_frequency,optimizer=optimizer,model_dir=model_dir_2,tensorboard=True,graph_conv_layers=graph_conv_layers)
#model_3=tyc_3DGCN_Model(mode=mode,n_tasks=n_tasks,batch_size=bs,predictor_hidden_feats=pred_h_feats,log_frequency=log_frequency,optimizer=optimizer,model_dir=model_dir_3,tensorboard=True,regularization_loss=l2_norm)

# Assuming 'model' is your DeepChem model


total_parameters_1 = sum(p.numel() for p in model_1.model.parameters() if p.requires_grad)
print(f"Total Parameters model 1: {total_parameters_1}")
#total_parameters_2 = sum(p.numel() for p in model_2.model.parameters() if p.requires_grad)
#print(f"Total Parameters model 2: {total_parameters_2}")
#total_parameters_3 = sum(p.numel() for p in model_3.model.parameters() if p.requires_grad)
#print(f"Total Parameters model 3: {total_parameters_3}")

#valid_model_3=ValidationCallback(dataset=valid,interval=5*log_frequency,metrics=[metric],save_dir=model_dir_3+"\\BEST_VAL",save_on_minimum=False,transformers=transformers)
#valid_model_2=ValidationCallback(dataset=valid,interval=5*log_frequency,metrics=[metric],save_dir="E:\WS2324\seminar\Tyc_easy_3DGCN\BEST_VAL",save_on_minimum=False,transformers=transformers)
valid_model_1=ValidationCallback(dataset=valid,interval=5*log_frequency,metrics=[metric],save_dir=model_dir_1+"\\BEST_VAL",save_on_minimum=False,transformers=transformers)


#TRAIN PART CODE commentize this part if you want pure test
print("fit begin")# by first time training please set restore=false
model_1.fit(train,nb_epoch=num_epoch,checkpoint_interval=checkpoint_interval,restore=False,max_checkpoints_to_keep=max_checkpoints_to_keep,all_losses=all_losses_1,callbacks=[valid_model_1])
print("fit-1 finished")
#model_2.fit(train,nb_epoch=num_epoch,checkpoint_interval=checkpoint_interval,restore=True,max_checkpoints_to_keep=max_checkpoints_to_keep,all_losses=all_losses_2)
print("fit-2 finished")
#model_3.fit(train,nb_epoch=num_epoch,checkpoint_interval=checkpoint_interval,restore=True,max_checkpoints_to_keep=max_checkpoints_to_keep,callbacks=[valid_model_3])
print("fit-3 finished")
print("fit end")

"""
if i_changed_hyper_params_above:
    if isinstance(scheduler,optimizers.LearningRateSchedule):
        d={'time_stamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'num_epochs':num_epoch,'batch_size':bs,'log_freq':log_frequency,'pred_h_feats':pred_h_feats,'graph_conv_layers':graph_conv_layers,'scheduler':scheduler.__class__.__name__,'optimizer':optimizer.__class__.__name__}
    else:
        d={'time_stamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'num_epochs':num_epoch,'batch_size':bs,'log_freq':log_frequency,'pred_h_feats':pred_h_feats,'graph_conv_layers':graph_conv_layers,'scheduler':scheduler,'optimizer':optimizer.__class__.__name__}
else:
    d={'time_stamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

with open(model_dir_1+'\loss_list.txt', 'a') as file:
    file.write(str(d))
    for item in all_losses_1:
        file.write("%s\n" % item)
with open(model_dir_2+'\loss_list.txt', 'a') as file:
    file.write(str(d))
    for item in all_losses_2:
        file.write("%s\n" % item)
with open(model_dir_3+'\loss_list.txt', 'a') as file:
    file.write(str(d))
    for item in all_losses_3:
        file.write("%s\n" % item)

print("model 1 training set score:",model_1.evaluate(train,[metric],transformers))
print("model 1 validate set score:",model_1.evaluate(valid,[metric],transformers))
#print("model 2 training set score:",model_2.evaluate(train,[metric],transformers))
#print("model 2 validate set score:",model_2.evaluate(valid,[metric],transformers))
#print("model 3 training set score:",model_3.evaluate(train,[metric],transformers))
#print("model 3 validate set score:",model_3.evaluate(valid,[metric],transformers))

"""


#TEST PART CODE If you want pure test, please commentize #TRAIN PART above
print("TEST TIME SCORE")
model_1.restore(model_dir=model_dir_1+"\\BEST_VAL")
print("model 1 test set score:",model_1.evaluate(test,[metric],transformers))
#model_2.restore()
#print("model 2 test set score:",model_2.evaluate(test,[metric],transformers))
#model_3.restore()
#print("model 3 test set score:",model_3.evaluate(test,[metric],transformers))










