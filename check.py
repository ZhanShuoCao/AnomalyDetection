import sys
sys.path.insert(0, '.')
from models.icdit_defect_generator import ICDiTDefectGenerator
from diffusion.scheduler import NoiseScheduler

config = {
    'model': {
        'text_encoder_name': 'google/t5-v1_1-base',       # matches config YAML
        'text_max_length': 512,
        'vae_name': 'stabilityai/sd-vae-ft-mse',
        'visual_encoder_name': 'dinov2_vitb14',
        'hidden_dim': 768, 'num_layers': 12, 'num_heads': 12,
        'patch_size': 2, 'latent_channels': 4,
        'training_mode': 'full_generator',
        'visual_embedding_source': 'reference_normal',
        'dedicated_layout_vae': True,                      # paper Fig.3
        'freeze_text_encoder': True, 'freeze_vae': True,
        'freeze_layout_encoder': True, 'freeze_visual_encoder': True,
    }
}
model = ICDiTDefectGenerator(config, image_size=256)
print('Model built successfully!')
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

# Verify forward() returns 7 values
import torch
dummy_latent = torch.randn(2, 4, 32, 32)
dummy_t = torch.randint(0, 1000, (2,))
dummy_mask = torch.zeros(2, 256, 256)
dummy_mask[0, 60:100, 80:120] = 1.0  # one defect region
dummy_ref = torch.randn(2, 3, 256, 256)
result = model.forward(
    noisy_latents=dummy_latent,
    timesteps=dummy_t,
    prompts=["test prompt A", "test prompt B"],
    masks=dummy_mask,
    reference_images=dummy_ref,
)
eps_pred, text_upd, layout_upd, visual_upd, layout_logits, text_init, visual_init = result
print(f'\nForward() output check:')
print(f'  eps_pred:       {tuple(eps_pred.shape)}')
print(f'  text_upd:       {tuple(text_upd.shape)}')
print(f'  layout_upd:     {tuple(layout_upd.shape)}')
print(f'  visual_upd:     {tuple(visual_upd.shape)}')
print(f'  layout_logits:  {tuple(layout_logits.shape)}')
print(f'  text_init:      {tuple(text_init.shape)}')
print(f'  visual_init:    {tuple(visual_init.shape)}')
print(f'\nAll checks passed!')