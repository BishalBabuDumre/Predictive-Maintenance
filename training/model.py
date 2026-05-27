import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4, hidden_layers=[32, 16], activation="LeakyReLU", dropout=None, beta=1.0, learning_rate = 0.001):
        super(VAE, self).__init__()

        self.beta = beta
        # Map activation strings to PyTorch modules
        activation_map = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "Tanh": nn.Tanh
        }
        act_fn = activation_map.get(activation, nn.LeakyReLU)

        # --- Dynamic Encoder Construction ---
        encoder_modules = []
        current_dim = input_dim
        for h_dim in hidden_layers:
            encoder_modules.append(nn.Linear(current_dim, h_dim))
            encoder_modules.append(act_fn())
            # NEW: Add dropout if configured
            if dropout is not None:
                encoder_modules.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Bottleneck layers remain tied to the final hidden layer dimension
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)
        
        # --- Dynamic Decoder Construction ---
        decoder_modules = []
        current_dim = latent_dim
        # Reverse the hidden layers list to reconstruct the mirror image
        for h_dim in reversed(hidden_layers):
            decoder_modules.append(nn.Linear(current_dim, h_dim))
            decoder_modules.append(act_fn())
            # NEW: Add dropout if configured
            if dropout is not None:
                decoder_modules.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        decoder_modules.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# FIXED: Integrated the beta scaling factor directly into the loss calculation
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (beta * KLD)
