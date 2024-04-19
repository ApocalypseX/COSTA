import torch
from torch import nn

#Simple MLP Encoder, use average as the final context encode
class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim = None,
        activation = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim), nn.Tanh()]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, query_trj: torch.Tensor=None, mean=True) -> torch.Tensor:
        if mean:
            return self.model(x).mean(-2)
        else:
            return self.model(x)

class MLPUDEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim = None,
        activation = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]

        self.output_dim = hidden_dims[-1]
        # if output_dim is not None:
        #     model += [nn.Linear(hidden_dims[-1], output_dim), nn.Tanh()]
        #     self.output_dim = output_dim
        self.mu=nn.Linear(hidden_dims[-1], output_dim)
        self.sigma=nn.Linear(hidden_dims[-1], output_dim)
        self.model = nn.Sequential(*model)
        self.softplus=nn.Softplus()

    def forward(self, x: torch.Tensor, query_trj: torch.Tensor=None, mean=False) -> torch.Tensor:
        x=self.model(x)
        mu=self.mu(x)
        sigma_squared=self.softplus(self.sigma(x))
        return mu, sigma_squared


#Simple RNN Encoder, use the last hidden state to encode the context
class RNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_hidden_dim,
        hidden_dims,
        output_dim = None,
        rnn_layer = 2,
        activation = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_dims = [rnn_hidden_dim] + list(hidden_dims)
        model = []
        sigmoid=nn.Sigmoid()
        self.rnn=nn.LSTM(input_dim,rnn_hidden_dim,rnn_layer,batch_first=True)
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim), nn.Tanh()]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, query_trj: torch.Tensor=None) -> torch.Tensor:
        out,(x,c)=self.rnn(x)
        x=x[-1]
        return self.model(x)

class MLPAttnEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]

        #model += [nn.Linear(hidden_dims[-1], output_dim), nn.Tanh()]
        self.out=nn.Linear(hidden_dims[-1], output_dim)
        self.gate=nn.Linear(hidden_dims[-1], 1)
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=1)
        self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, query_trj: torch.Tensor=None) -> torch.Tensor:
        x=self.model(x)
        b,s,dim=x.shape
        key=self.gate(x).reshape(b,s)
        scores=self.softmax(key)
        res=scores.unsqueeze(2).expand_as(x).mul(x).sum(1)
        return self.tanh(self.out(res))

#Self Attention Encoder, use one trajectory and the rnn to get the query, and all transitions to form keys and values
class SelfAttnEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        attn_dim,
        hidden_dims,
        output_dim = None,
        rnn_layer = 2,
        num_heads = 2,
        activation = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_dims = [attn_dim] + list(hidden_dims)
        model = []
        sigmoid=nn.Sigmoid()
        self.query_rnn=nn.LSTM(input_dim,attn_dim,rnn_layer,batch_first=True)
        self.key=nn.Linear(input_dim,attn_dim)
        self.value=nn.Linear(input_dim,attn_dim)
        self.mha=nn.MultiheadAttention(attn_dim,num_heads)
        self.relu=nn.ReLU()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim), nn.Tanh()]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

        for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                 nn.init.constant_(m.bias, 0)
            #  elif isinstance(m, nn.LSTM):
            #      nn.init.orthogonal(m.weight, gain=1)
            #      nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor, query_trj: torch.Tensor) -> torch.Tensor:
        out,(query,c)=self.query_rnn(x)
        query=query[-1]
        key=self.relu(self.key(x))
        value=self.relu(self.value(x))
        bs=key.shape[0]
        ls=key.shape[1]
        query=query.reshape(1,bs,-1)
        key=key.reshape(ls,bs,-1)
        value=value.reshape(ls,bs,-1)
        attn_out,_=self.mha(query,key,value)
        return self.model(attn_out[0])