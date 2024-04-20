from tyc_3DGCN_Model import tyc_3DGCN_Model
from tyc_2DGCN_Model import tyc_2DGCN_Model
import deepchem.molnet as dcmolnet
from tyc_root_featurizer import tyc_MolGraphConvFeaturizer
from deepchem.models import optimizers
from deepchem.models.callbacks import ValidationCallback
from deepchem.metrics import pearson_r2_score
from deepchem.metrics import Metric
import math
import matplotlib.pyplot as plt


def l2_norm(self):
    lambda_reg=0.00001
    l2_penalty = sum(p.pow(2.0).sum() for p in self.model.parameters())
    return lambda_reg * l2_penalty

tasks,dataset,transformers=dcmolnet.load_qm7(featurizer=tyc_MolGraphConvFeaturizer(),reload=False)
train,valid,test=dataset
model_dir="E:\\WS2324\\FINAL\\3D\\pred_h_feats=128"
learning_rate=0.001
optimizer=optimizers.Adam(learning_rate=learning_rate)
metric = Metric(pearson_r2_score)#Metric


checkpoint_interval=10
max_checkpoints_to_keep=200
num_epoch=100
bs=9000
pred_h_f=128
log_frequency=math.ceil(4000/bs)
steps_per_epoch=math.ceil(5470/bs)
patience_in_step=20*steps_per_epoch
patience=math.ceil(patience_in_step/(3*log_frequency))


valid_model_3d=ValidationCallback(dataset=valid,interval=3*log_frequency,metrics=[metric],save_dir=model_dir+"\\BEST_VAL",save_on_minimum=False,transformers=transformers,patience=patience)
model_3d=tyc_3DGCN_Model(mode="regression",n_tasks=len(tasks),batch_size=bs,predictor_hidden_feats=pred_h_f,
                                 log_frequency=log_frequency,optimizer=optimizer,model_dir=model_dir,tensorboard=True,regularization_loss=l2_norm
                                 )

print("training 3d model for hidden feats %d,batch size:%d,learning rate:%d,patience:%d" %(pred_h_f,bs,learning_rate,patience))
#model_3d.fit(train,nb_epoch=num_epoch,checkpoint_interval=checkpoint_interval,restore=True,
                     #max_checkpoints_to_keep=max_checkpoints_to_keep,callbacks=[valid_model_3d])

print("3d finished")
model_3d.restore()
score=model_3d.evaluate(test,[metric],transformers)
print(score)
print("model test set score:",score)
pred=model_3d.predict(test,transformers=transformers)
y=test.y
#print(pred)
print("_____________")
#print(y)
a=[]
b=[]
for i in range(len(pred)):
   a.append(pred[i][0])
   b.append(y[i][0])

plt.scatter(a, b)
plt.title("prediction and ground truth with pearson r2 score %f" %score["pearson_r2_score"])
plt.xlabel('predictions')
plt.ylabel('ground truths')
plt.show()

