import torch

from model import GatedLinearAttention, GLAConfig

if __name__ == "__main__":

    BATCH, H, N_CTX, D_MODEL = 32, 4, 2048, 1024

    config = GLAConfig(d_model=D_MODEL, n_head=H)
    print(config)

    GLA = GatedLinearAttention(config).cuda().to(torch.bfloat16)

    x = torch.randn((BATCH, N_CTX, D_MODEL),
                    dtype=torch.bfloat16,
                    device="cuda",
                    requires_grad=False)

    y, _ = GLA(x)
    print(y.shape)
