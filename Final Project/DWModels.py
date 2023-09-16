import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import (
    Any,
    Dict,
    List,
    Match,
    Tuple,
    Union,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Generator,
    Collection,
)


from torch import Tensor
allowed_activations = [
    "relu",
    "leaky_relu",
    "softplus",
]


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if activation == "softplus":
        return nn.Softplus()


def dense_layer(
    inp: int,
    out: int,
    activation: str,
    p: float,
    bn: bool,
    linear_first: bool,
):
    act_fn = get_activation(activation)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]  #batch norm has bias itself
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        batchnorm: bool,
        batchnorm_last: bool,
        linear_first: bool,
        dropout: List[float] = None,
    ):
        super(MLP, self).__init__()

        if dropout is None:
            dropout = [0.0] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)


class CatEmbeddingsAndCont(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float,
        continuous_cols: Optional[List[str]],
    ):
        super(CatEmbeddingsAndCont, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        # Categorical
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict(
                {
                    "emb_layer_" + col: nn.Embedding(val + 1, dim, padding_idx=0) for col, val, dim in self.embed_input
                }
            )
            self.embedding_dropout = nn.Dropout(embed_dropout)
            self.emb_out_dim: int = int(
                np.sum([embed[2] for embed in self.embed_input])
            )
        else:
            self.emb_out_dim = 0

        # Continuous
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            self.cont_out_dim = len(self.continuous_cols)
            self.cont_norm = nn.BatchNorm1d(self.cont_out_dim)
        else:
            self.cont_out_dim = 0

        self.output_dim = self.emb_out_dim + self.cont_out_dim

    def forward(self, X: Tensor) -> Tuple[Tensor, Any]:
        if self.embed_input is not None:
            embed = [
                self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long()) for col, _, _ in self.embed_input
            ]
            x_emb = torch.cat(embed, 1)
            x_emb = self.embedding_dropout(x_emb)
        else:
            x_emb = None
        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
        else:
            x_cont = None

        return x_emb, x_cont

class TabMlp(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int, int]]] = None,
        embed_dropout: float = 0.1,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = "batchnorm",
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "relu",
        mlp_dropout: List[float] = None,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = False,
        pred_dim: int = 1,
    ):
        super(TabMlp, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.mlp_hidden_dims = mlp_hidden_dims
        self.embed_dropout = embed_dropout
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_linear_first = mlp_linear_first
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.pred_dim = pred_dim

        # CatEmbeddingsAndCont
        self.cat_embed_and_cont = CatEmbeddingsAndCont(
            column_idx,
            embed_input,
            embed_dropout,
            continuous_cols,
        )

        # MLP
        mlp_input_dim = self.cat_embed_and_cont.output_dim
        mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims
        self.tab_mlp = MLP(
            d_hidden=mlp_hidden_dims,
            activation = self.mlp_activation,
            dropout = self.mlp_dropout,
            batchnorm = self.mlp_batchnorm,
            batchnorm_last = self.mlp_batchnorm_last,
            linear_first = self.mlp_linear_first,
        )


        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = mlp_hidden_dims[-1]

        self.pred_layer = nn.Linear(self.output_dim, self.pred_dim)

    def forward(self, X: Tensor) -> Tensor:
        x_emb, x_cont = self.cat_embed_and_cont(X)
        if x_emb is not None:
            x = x_emb
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_emb is not None else x_cont
        return self.pred_layer(self.tab_mlp(x))
        
class DeepWide(nn.Module):
    def __init__(self,cat_F_dims,emmbeded_hyper,MLP_dims,numerical_dim):
        super(DeepWide,self).__init__()
        cat_dims = sum(cat_F_dims)
        self.embeding = nn.Embedding(cat_dims, emmbeded_hyper)
        input_dims = numerical_dim + len(cat_F_dims) * emmbeded_hyper
        self.FXlayer = nn.Sequential(nn.Linear(input_dims, 1),nn.ReLU())
        modules = []
        for output in MLP_dims:
            modules.append(nn.Linear(input_dims, output))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.1))
            input_dims = output
        self.MLP = nn.Sequential(*modules)
        self.Flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x1,x2):
        embedded_output = torch.Tensor(self.embeding(x1.to(torch.int64)))
        square_of_sum = torch.sum(embedded_output, axis=1) ** 2
        sum_of_square = torch.sum(embedded_output ** 2, axis=1)
        embedded_output2 = self.Flatten(embedded_output)
        cated_input = torch.cat((embedded_output2, x2), -1 )
        z = self.FXlayer(cated_input)
        z+=self.MLP(cated_input)
        z+=0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        return self.sigmoid(z).reshape(-1)
    
class DeepWide2(nn.Module):
    def __init__(self,cat_F_dims,emmbeded_hyper,MLP_dims,numerical_dim):
        super(DeepWide2,self).__init__()
        cat_dims = sum(cat_F_dims)
        self.embeding = nn.Embedding(cat_dims, emmbeded_hyper)
        input_dims = len(cat_F_dims) * emmbeded_hyper
        modules = []
        for output in MLP_dims:
            modules.append(nn.Linear(input_dims, output))
            modules.append(nn.ReLU())
            input_dims = output
        self.MLP = nn.Sequential(*modules)
        self.Flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        input_dims+= numerical_dim #+ len(cat_F_dims) * emmbeded_hyper
        self.FXlayer = nn.Sequential(nn.Linear(input_dims, 1),nn.ReLU())
    def forward(self,x1,x2):
        embedded_output = torch.Tensor(self.embeding(x1.to(torch.int64)))
        square_of_sum = torch.sum(embedded_output, axis=1) ** 2
        sum_of_square = torch.sum(embedded_output ** 2, axis=1)
        embedded_output2 = self.Flatten(embedded_output)
        #cated_input = torch.cat((embedded_output2, x2), -1 )
        x3 = self.MLP(embedded_output2)
        cated_input2 = torch.cat((x3,x2), -1 )
        z = self.FXlayer(cated_input2)
        z+=0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        return self.sigmoid(z).reshape(-1)
       
