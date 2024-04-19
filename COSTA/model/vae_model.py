from torch import nn
import torch

class VAELoss(nn.Module):
    def __init__(self, decode_type="CVAE", beta=1.0):
        super().__init__()
        self.decode_type=decode_type
        self.criterion=nn.MSELoss(reduction='sum')
        self.beta=beta

    def recon_loss(self, x, x_):
        return self.criterion(x_, x)

    def kl_div(self, mu, logv):
        kld_loss=-0.5*torch.sum(1+logv-mu**2-torch.exp(logv))
        return kld_loss

    def forward(self, x, x_, mu=-1, logv=-1):
        recon_loss=self.recon_loss(x, x_)
        if self.decode_type=="AE":
            return {"recon_loss":recon_loss}
        kld_loss=self.kl_div(mu, logv)
        total_loss=recon_loss+self.beta*kld_loss
        return {"total_loss":total_loss, "recon_loss":recon_loss, "kld_loss":kld_loss}

class CVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action=1., hidden_unit=128):
        super(CVAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.e2 = nn.Linear(hidden_unit, hidden_unit)

        self.mean = nn.Linear(hidden_unit, latent_dim)
        self.log_std = nn.Linear(hidden_unit, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_unit)
        self.d2 = nn.Linear(hidden_unit, hidden_unit)
        self.d3 = nn.Linear(hidden_unit, action_dim)
        self.relu = nn.ReLU()

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, action, state):
        z = self.relu(self.e1(torch.cat([state, action], 1)))
        z = self.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        log_v = 2*log_std
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, log_v

    def decode(self, state, z=None):
        if z is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            z = torch.randn((state.shape[0], self.latent_dim)).to(device).clamp(-0.5, 0.5)
        a = self.relu(self.d1(torch.cat([state, z], 1)))
        a = self.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))