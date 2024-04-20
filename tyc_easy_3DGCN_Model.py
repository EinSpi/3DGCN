from deepchem.models.torch_models.torch_model import TorchModel
from tyc_easy_3DGCN_Core import tyc_easy_3DGCN_Core
from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
import torch
import torch.nn.functional as F

class tyc_easy_3DGCN_Model(TorchModel):
    def __init__(self,
               n_tasks: int=1,
               graph_conv_layers: list = None,
               activation=F.relu,
               residual: bool = True,
               batchnorm: bool = False,
               dropout: float = 0.,
               predictor_hidden_feats: int = 128,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features=30,
               n_classes: int = 2,
               self_loop: bool = True,
               
               **kwargs):
        
        model = tyc_easy_3DGCN_Core(
        n_tasks=n_tasks,
        graph_conv_layers=graph_conv_layers,
        activation=activation,
        residual=residual,
        batchnorm=batchnorm,
        dropout=dropout,
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout,
        mode=mode,
        number_atom_features=number_atom_features,
        n_classes=n_classes,
        )



        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']

        
        super(tyc_easy_3DGCN_Model, self).__init__(model, loss=loss, output_types=output_types, **kwargs)

        self._self_loop = self_loop
        self.number_atom_features=number_atom_features
        

    def _prepare_batch(self, batch):
    
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch
        dgl_graphs=[]
        
        for graph in inputs[0]:
            
            dgl_graph=graph.to_dgl_graph(self_loop=self._self_loop)
            dgl_graph.ndata['v'] = torch.zeros(dgl_graph.number_of_nodes(),3,self.number_atom_features).float()
            dgl_graphs.append(dgl_graph)
        
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(tyc_easy_3DGCN_Model, self)._prepare_batch(([], labels,
                                                               weights))
        return inputs, labels, weights
    
 
        