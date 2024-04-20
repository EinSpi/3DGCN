from tyc_3DGCN_Model import tyc_3DGCN_Model
from tyc_2DGCN_Model import tyc_2DGCN_Model
import deepchem.molnet as dcmolnet
from tyc_root_featurizer import tyc_MolGraphConvFeaturizer
from deepchem.models import optimizers
from deepchem.models.callbacks import ValidationCallback
from deepchem.metrics import pearson_r2_score
from deepchem.metrics import Metric
import math

"""
def l2_norm(self):
    lambda_reg=0.00001
    l2_penalty = sum(p.pow(2.0).sum() for p in self.model.parameters())
    return lambda_reg * l2_penalty
"""

tasks,dataset,transformers=dcmolnet.load_qm7(featurizer=tyc_MolGraphConvFeaturizer(),reload=False)
train,valid,test=dataset
metric = Metric(pearson_r2_score)#Metric


pred_hidden_feats=[32,64,128]
batch_sizes=[32,64,128,256,512,1024,2048,4096,5470]
learning_rates=[0.002,0.001]
checkpoint_interval=10
max_checkpoints_to_keep=200
num_epoch=100

lr_bs=[[32,0.002],[64,0.002],[128,0.001],[256,0.001],[512,0.001],[1024,0.001],[2048,0.001],[4096,0.001],[5470,0.001]]

model_dir=[]
for pred_h_f in pred_hidden_feats:
    model_dir.append("E:\\WS2324\\FINAL\\2D\\pred_h_feats=%d" %pred_h_f)
for pred_h_f in pred_hidden_feats:
    model_dir.append("E:\\WS2324\\FINAL\\3D_wo_regular_2layers\\pred_h_feats=%d" %pred_h_f)

for index,lb in enumerate(lr_bs):
    learning_rate=lb[1]
    batch_size=lb[0]
    log_frequency=math.ceil(4000/batch_size)
    steps_per_epoch=math.ceil(5470/batch_size)
    patience_in_step=20*steps_per_epoch
    patience=math.ceil(patience_in_step/(3*log_frequency))
    print("patience:%d" %patience)
    optimizer=optimizers.Adam(learning_rate=learning_rate)
    for i,pred_h_f in  enumerate(pred_hidden_feats):

        #model_2d=tyc_2DGCN_Model(mode="regression",n_tasks=len(tasks),batch_size=batch_size,predictor_hidden_feats=pred_h_f,
                                #log_frequency=log_frequency,optimizer=optimizer,model_dir=model_dir[i],tensorboard=True,
                                         #graph_conv_layers=[64,30],batchnorm=True,regularization_loss=l2_norm)
        
        model_3d=tyc_3DGCN_Model(mode="regression",n_tasks=len(tasks),batch_size=batch_size,predictor_hidden_feats=pred_h_f,
                                 log_frequency=log_frequency,optimizer=optimizer,model_dir=model_dir[i+3],tensorboard=True
                                 )
        
        #valid_model_2d=ValidationCallback(dataset=valid,interval=3*log_frequency,metrics=[metric],save_dir=model_dir[i]+"\\BEST_VAL",save_on_minimum=False,transformers=transformers,patience=patience)
        valid_model_3d=ValidationCallback(dataset=valid,interval=3*log_frequency,metrics=[metric],save_dir=model_dir[i+3]+"\\BEST_VAL",save_on_minimum=False,transformers=transformers,patience=patience)

        print("training 2d model for hidden feats %d,batch size:%d,learning rate:%d,patience:%d" %(pred_h_f,batch_size,learning_rate,patience))
        #model_2d.fit(train,nb_epoch=num_epoch,checkpoint_interval=checkpoint_interval,restore=False if index==0 else True,
                     #max_checkpoints_to_keep=max_checkpoints_to_keep,callbacks=[valid_model_2d])
        print("2d finished")
        print("training 3d model for hidden feats %d,batch size:%d,learning rate:%d,patience:%d" %(pred_h_f,batch_size,learning_rate,patience))
        model_3d.fit(train,nb_epoch=num_epoch,checkpoint_interval=checkpoint_interval,restore=False if index==0 else True,
                     max_checkpoints_to_keep=max_checkpoints_to_keep,callbacks=[valid_model_3d])
        print("3d finished")
        
