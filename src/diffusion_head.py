import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_layers import Conv1dBlock, ConditionalResidualBlock1D, Downsample1d, Upsample1d, SinusoidalPosEmb, ConditionalUnet1D

class UNetConv(torch.nn.Module):
    def __init__(self, image_emb_dim, 
                 state_emb_dim, 
                 traj_dim, 
                 global_condn_dim,
                 use_residuals = False,
                 diffusion_step_embed_dim=256, 
                 down_dims=[256, 512, 1024],
                 kernel_size=5, 
                 n_groups=8):
        
        self.in_emb_dim = image_emb_dim + state_emb_dim
        self.traj_embed_dim = traj_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims
        self.use_residuals = use_residuals

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = torch.nn.Sequential(
            SinusoidalPosEmb(dsed),
            torch.nn.Linear(dsed, dsed * 4),
            torch.nn.Mish(),
            torch.nn.Linear(dsed * 4, dsed),
        )
        
        cond_dim = dsed + global_condn_dim

        # Create in_out pairs for downsampling layers
        in_out = list(zip(self.in_emb_dim[:-1], self.in_emb_dim[1:]))
        
        # dimension of the intermediate representation
        mid_dim = self.in_emb_dim[-1]
        
        down_modules = torch.nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                torch.nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            
            # Modules in the middle
        self.mid_modules = torch.nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )
        
        
        

class StateEmbedding(torch.nn.Module):
    def __init__(self, car_fwd_speed, steering_angle, embedding_dim) -> None:
        super(StateEmbedding, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        

class DiffusionHead(torch.nn.Module):
    def __init__(self, vision_embedding_dim, state_embedding_dim) -> None:
        self.vision_embedding_dim = vision_embedding_dim
        self.state_embedding_dim = state_embedding_dim
        
        self.unet_conv = UnetConv()
        
    def forward(self, vision_embeddings: torch.Tensor, state_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_embeddings: Tensor of shape (batch_size, vision_embedding_dim)
            state_embeddings: Tensor of shape (batch_size, state_embedding_dim)
        
        Returns:
            combined_embeddings: Tensor of shape (batch_size, vision_embedding_dim + state_embedding_dim)
        """
        combined_embeddings = torch.cat((vision_embeddings, state_embeddings), dim=-1)

        
        
        
        
    
        