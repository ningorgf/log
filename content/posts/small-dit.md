+++
title = 'Small Dit'
date = 2024-03-13T13:58:16+08:00
draft = false
+++

Attempt to replicate a mini DiT model. While studying a paper, I found it challenging to comprehend the Loss, so I decided to explore the code implementation of loss. The code is based on some intros from CS294-158-SP24 Deep Unsupervised Learning Spring 2024 https://sites.google.com/view/berkeley-cs294-158-sp24/home. (I recommend checking out their YouTube course, which covers a lot of the latest papers and interpretations. The homeworks also offer a great deal of freedom, allowing you to be quite creative.)


### VAE
For a detailed introduction, refer to the [wiki](https://en.wikipedia.org/wiki/Variational_autoencoder).

![f1](../images/f1.png)
Fig1：[resourse](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

$z=\mu_{\phi}(x)+L_{\phi}(x)\epsilon$

Reparametrisation trick allows the error to be backpropagated through the network.

Although this homework requires encoding to generate z with the dimensions of (B, 4, 8, 8), I modified it in the code to be (B, 2 * 4, 8, 8).

```
class VAE(nn.Module):
    def __init__(self):


    def encode(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim=1)  # Split the encoded values into mu and log_var
        return self.reparameterize(mu, log_var), mu, log_var


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), mu, log_var


    @property
    def latent_shape(self):
        return (4, 8, 8)


# Loss function
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

```
### LDM

![f2](../images/f2.png)
Fig2. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf) 

### DiT
[Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748.pdf)
These Diffusion Transformers (DiTs), replace the commonly-used U-Net backbone with a transformer that operates on latent patches of images. The authors conduct an analysis of the scalability of their DiTs by examining the forward pass complexity, measured in Giga FLoating-point Operations Per Second (GFlops). It was found that DiTs with a higher number of GFlops - achieved through increased transformer depth/width or an escalated number of input tokens - consistently have lower Fréchet Inception Distance (FID), a measure of similarity between sets of images. In addition to their scalability, the largest of these DiTs ('DiT-XL/2') outperform all prior diffusion models on 512x512 and 256x256 class-conditional ImageNet benchmarks, achieving a state-of-the-art FID of 2.27 on the latter.


![f3](../images/dit.png)
Fig 3: The Diffusion Transformer (DiT) architecture. Left: We train conditional latent DiT models. The input latent is decomposed into patches and processed by several DiT blocks. Right: Details of our DiT blocks. We experiment with variants of standard transformer blocks that incorporate conditioning via adaptive layer norm, cross-attention and extra input tokens. Adaptive layer norm works best.

The connection between portions of the pseudocode and Figure 3:
![f4](../images/editDit.png)
```
DiT(input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes, cfg_dropout_prob)
    Given x (B x C x H x W) - image, y (B) - class label, t (B) - diffusion timestep
    x = patchify_flatten(x) # B x C x H x W -> B x (H // P * W // P) x D, P is patch_size
    x += pos_embed # see get_2d_sincos_pos_embed

    t = compute_timestep_embedding(t) # Same as in UNet
    if training:
        y = dropout_classes(y, cfg_dropout_prob) # Randomly dropout to train unconditional image generation
    y = Embedding(num_classes + 1, hidden_size)(y) # y = c (B x D)
    c = t + y #c (B x D)

    for _ in range(num_layers):
        x = DiTBlock(hidden_size, num_heads)(x, c)  # x (B x L x D), c (B x D)

    x = FinalLayer(hidden_size, patch_size, out_channels)(x)
    x = unpatchify(x) # B x (H // P * W // P) x (P * P * 2C) -> B x 2C x H x W
    return x
```

Based on the pseudocode, I've constructed a mini DiT; the code is at the end of the document. This code isn't universally executable. For the full version of DiT, please refer to the paper's [GitHub](https://github.com/facebookresearch/DiT/tree/main).




####  Loss
The structure of the input and output are not the same. As mentioned in the paper, (reparameterize $\mu_{\phi}$), the model can be trained using a simple mean-squared error between the predicted noise $\epsilon_{\theta}(x_{t})$ and the sampled ground truth Gaussian noise $\epsilon_{t}$ (the z in the VAE).

However, to train diffusion models with a learned reverse process covariance, the full $D_{KL}$ term needs to be optimized.The following source code may help in understanding the loss:

```
DiT/diffusion/diffusion_utils.py、
def normal_kl(mean1, logvar1, mean2, logvar2):
 return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

DiT/diffusion/gaussian_diffusion.py
assert model_output.shape == target.shape == x_start.shape
terms["mse"] = mean_flat((target - model_output) ** 2)
if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

p_sample
 sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
 return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

### code
the final layer's out_channels is 2C.
```
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim // 2) / 2.0
    omega = 1.0 / (10000 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = torch.arange(grid_size).float()
    grid_w = torch.arange(grid_size).float()
    grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

# Example usage:
embed_dim = 768  # Example embedding dimension
grid_size = 16   # Example grid size
pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
# get_2d_sincos_pos_embed(C * patch_size * patch_size, H // patch_size).reshape(1, H // patch_size * W // patch_size, -1).repeat(B, 1, 1)
get_2d_sincos_pos_embed(2 * 4 * 4, 12 // 4).reshape(1, 3 * 3, -1).shape

```
```
# @title DiTBlock

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(DiTBlock, self).__init__()
        # Define the layers used in DiTBlock
        self.silu = nn.SiLU()
        self.linear_c = nn.Linear(hidden_size, 6 * hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        # Define an MLP. Adjust the architecture as needed.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, c):
        # Process context vector
        c = self.silu(c)
        c = self.linear_c(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)

        # First half of the DiTBlock (MSA)
        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        attn_output, _ = self.attention(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # Second half of the DiTBlock (MLP)
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        mlp_output = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x
```

```
# @title FinalLayer

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        # Define the layers used in FinalLayer
        self.silu = nn.SiLU()
        self.linear_c = nn.Linear(hidden_size, 2 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, c):
        # Process context vector
        c = self.silu(c)
        c = self.linear_c(c)
        shift, scale = c.chunk(2, dim=1)

        # Apply LayerNorm and Modulation
        x = self.norm(x)
        x = modulate(x, shift, scale)

        # Apply final linear layer
        x = self.final_linear(x)
        return x


```
```

def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    device = timesteps.device
    freqs = torch.exp(-torch.log(torch.tensor(max_period, device=device)) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    args = timesteps[:, None].to(device=device, dtype=torch.float32) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # If dim is odd, pad with a column of zeros
        embedding = F.pad(embedding, (0, 1))
    return embedding
import torch

def patchify_flatten(imgs, patch_size):
    B, C, H, W = imgs.shape
    patch_flat_dim = C * patch_size * patch_size
    # Step 1: Reshape
    imgs = imgs.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    # Step 2: Transpose to B x num_patches x patch_flat_dim
    imgs = imgs.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, patch_flat_dim)
    return imgs


def unpatchify(x, patch_size, H, W):
    """
    Args:
    - x: the flattened patches tensor, shape (B, num_patches, patch_depth)
    - patch_size: the size of one side of the square patch
    - H: the height of the original image
    - W: the width of the original image

    Returns:
    - img: the reconstructed image tensor, shape (B, C, H, W)
    """
    B, num_patches, patch_depth = x.shape
    C = patch_depth // (patch_size * patch_size)

    # Step 1: Reshape to (B, H // patch_size, W // patch_size, patch_size, patch_size, C)
    x = x.reshape(B, H // patch_size, W // patch_size, patch_size, patch_size, C)

    # Step 2: Transpose to (B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 5, 1, 3, 2, 4)

    # Step 3: Reshape to the original image shape (B, C, H, W)
    img = x.reshape(B, C, H, W)

    return img
def dropout_classes(y, dropout_prob):
        # During training, randomly set some entries to -1 to indicate the special no-class case
    mask = torch.rand_like(y, dtype=torch.float32) < dropout_prob
    y[mask] = y_max+1;
    return y


class DiT(nn.Module):
    # Initialization remains the same as previously provided
    def __init__(self, input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes, cfg_dropout_prob):
        super(DiT, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cfg_dropout_prob = cfg_dropout_prob

        # Assuming C, H, W are the channels, height, and width of the input
        C, H, W = input_shape
        self.D = C * (patch_size ** 2)  # Depth of the flattened patches
        self.num_patches = (H // patch_size) * (W // patch_size)

        grid_size = max(H // patch_size, W // patch_size)
        self.pos_embed = get_2d_sincos_pos_embed(self.D , grid_size)

        # Embedding layer for class labels
        self.class_embedding = nn.Embedding(num_classes + 1, hidden_size)


        # Stack of DiTBlocks
        self.diT_blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])

        # Final layer to map to output space
        self.final_layer = FinalLayer(hidden_size, patch_size, 2 * C)

    def forward(self, x, y, t, training=True):
        B, C, H, W = x.shape

        # Patchify and flatten input images
        print(x.shape)
        x = patchify_flatten(x, self.patch_size)  # Needs to be implemented

        # Add positional embedding
        print(x.shape)

        x += self.pos_embed.reshape(1, H // self.patch_size * W // self.patch_size, -1).repeat(B, 1, 1)
        print(x.shape)
        # Compute timestep embedding using the PyTorch-based function
        t_emb = timestep_embedding(t, C * self.patch_size * self.patch_size)
        print(t_emb.shape)

        # Apply dropout to class labels if training
        if training:
            y = dropout_classes(y, self.cfg_dropout_prob)

        # Embed class labels
        y_emb = self.class_embedding(y)
        print('y_emb')
        print(y_emb.shape)

        # Combine timestep and class label embeddings
        c = t_emb + y_emb

        # Pass the input through DiTBlocks
        for diT_block in self.diT_blocks:
            x = diT_block(x, c)

        # Pass the output through the final layer
        x = self.final_layer(x, c)

        # Unpatchify the output to obtain the final image
        x = unpatchify(x, self.patch_size, H, W) 

        return x

```


