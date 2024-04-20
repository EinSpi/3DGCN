import torch

class tyc_batch_norm(torch.nn.Module):
    def __init__(self,
               number_atom_features=30,
               affine=False,
               **kwargs):
        super(tyc_batch_norm,self).__init__()
        self.batch_norm_s=torch.nn.BatchNorm1d(number_atom_features,affine=affine)
        self.batch_norm_v=torch.nn.BatchNorm1d(3*number_atom_features,affine=affine)

    def forward(self,feats:torch.Tensor , feats_3d:torch.Tensor):
        normed_feats=self.batch_norm_s(feats)
        normed_feats_3d=self.batch_norm_v(feats_3d.view(feats_3d.shape[0],-1))
        normed_feats_3d=normed_feats_3d.view(feats_3d.shape[0],3,-1)        
        return normed_feats, normed_feats_3d
