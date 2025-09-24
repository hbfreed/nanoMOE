"""Minimal comparison of MoeMLPMegaBlocks vs MoeMLPSTK"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange
import numpy as np
import stk
from megablocks import ops
from megablocks.layers.gelu import gelu
from model import MoeMLPMegaBlocks, MoeMLPSTK

# Create config
@dataclass
class TestConfig:
    n_embd: int = 128
    d_ffn: int = 256
    bias: bool = False
    dropout: float = 0.0
    num_experts: int = 8
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    block_size: int = 128
    block_k: int = 64
    n_ctx: int = 1024  # Added for STK model

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cpu':
    print("WARNING: MegaBlocks requires CUDA. Running anyway...")
print()

# def main():
#     torch.manual_seed(42)
#     # Config
#     config = TestConfig()

#     # Create models
#     model_megablocks = MoeMLPMegaBlocks(config).to(device)
#     model_stk = MoeMLPSTK(config).to(device)
#     model_stk.w1 = model_megablocks.w1
#     model_stk.w2 = model_megablocks.w2
#     model_stk.router = model_megablocks.router


#     # Create same input
#     batch_size = 2
#     seq_len = 1024
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)

#     # For loss calculation
#     target = torch.randn(batch_size, seq_len, config.n_embd, device=device)
#     criterion = nn.CrossEntropyLoss()

#     print("=" * 50)
#     print("MoE Comparison: MegaBlocks vs STK")
#     print("=" * 50)
#     print(f"Input shape: {x.shape}")
#     print(f"Config: experts={config.num_experts}, experts_per_tok={config.num_experts_per_tok}")
#     print()

#     # Run both models
#     with torch.no_grad():
#         out_mega, aux_mega, f_mega = model_megablocks(x)
#         out_stk, aux_stk, f_stk = model_stk(x)

#     # Compare outputs
#     print("Output comparison:")
#     print(f"  Output shape: {out_mega}")
#     print(f"  Output shape: {out_stk}")
#     print(f"  Forward diff (max): {(out_mega - out_stk).abs().max().item():.6e}")
#     print(f"  Forward diff (mean): {(out_mega - out_stk).abs().mean().item():.6e}")
#     print(torch.allclose(out_mega,out_stk))

#     # Compare losses
#     loss_mega = criterion(out_mega, target)
#     loss_stk = criterion(out_stk, target)
#     print("Loss comparison:")
#     print(f"  MegaBlocks loss: {loss_mega.item():.6f}")
#     print(f"  STK loss: {loss_stk.item():.6f}")
#     print(f"  Loss diff: {(loss_mega - loss_stk).abs().item():.6e}")
#     print()

#     # Compare auxiliary losses
#     print("Auxiliary loss comparison:")
#     print(f"  Router Z-loss diff: {(aux_mega['router_z_loss'] - aux_stk['router_z_loss']).abs().item():.6e}")
#     print(f"  Load balance diff: {(aux_mega['load_balance_loss'] - aux_stk['load_balance_loss']).abs().item():.6e}")
#     print()

#     # Compare expert usage
#     print("Expert usage comparison:")
#     print(f"  Expert usage diff (max): {(f_mega - f_stk).abs().max().item():.6e}")
#     print(f"  Expert usage diff (mean): {(f_mega - f_stk).abs().mean().item():.6e}")

# def compare_forward_passes():
#     config = TestConfig()
#     batch_size = 2
#     seq_len = 8
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)

#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)

#     # Copy weights to ensure they're identical
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)

#     batch_size, seq_len, n_embd = x.shape
#     x_flat = x.reshape(-1, n_embd)

#     # 1. Router outputs
#     router_logits_mb = mb_model.router(x_flat)
#     router_logits_stk = stk_model.router(x_flat)
#     print(f"router allclose: {torch.allclose(router_logits_mb, router_logits_stk)}")

#     # 2. Router probabilities and top-k
#     probs_mb = F.softmax(router_logits_mb, dim=-1, dtype=torch.float)
#     probs_stk = F.softmax(router_logits_stk, dim=-1, dtype=torch.float)
#     print(f"probs allclose: {torch.allclose(probs_mb, probs_stk)}")

#     weights_mb, experts_mb = torch.topk(probs_mb, config.num_experts_per_tok)
#     weights_stk, experts_stk = torch.topk(probs_stk, config.num_experts_per_tok)
#     print(f"weights allclose: {torch.allclose(weights_mb,weights_stk)}")
#     print(f"experts equal: {torch.equal(experts_mb,experts_stk)}")

#     # 3. Weight normalization
#     if config.norm_topk_prob:
#         weights_mb = weights_mb / weights_mb.sum(dim=-1, keepdim=True)
#         weights_stk = weights_stk / weights_stk.sum(dim=-1, keepdim=True)
#         print(f"norm_topk_prob allclose: {torch.allclose(weights_mb, weights_stk)}")

#     # MegaBlocks forward pass
#     with torch.no_grad():
#         # Flatten weights and experts
#         expert_weights_flat = weights_mb.reshape(-1)
#         selected_experts_flat = experts_mb.reshape(-1)

#         # Sort tokens by expert
#         bin_ids, indices, tokens_per_expert = mb_model._sort_tokens_by_expert(selected_experts_flat)

#         # Create topology
#         padded_bins, topology = mb_model._create_topology(x_flat, tokens_per_expert)

#         # Gather tokens
#         x_permuted = mb_model._gather_tokens(x_flat, indices, bin_ids, tokens_per_expert, padded_bins)

#         # Pad if needed
#         if x_permuted.shape[0] != topology.shape[0]:
#             if x_permuted.shape[0] < topology.shape[0]:
#                 padding = torch.zeros((topology.shape[0] - x_permuted.shape[0], x_permuted.shape[1]),
#                                      dtype=x_permuted.dtype, device=x_permuted.device)
#                 x_permuted = torch.cat([x_permuted, padding], dim=0)

#         # First MLP layer
#         x_after_w1 = stk.ops.sdd(x_permuted, mb_model.w1, topology)

#         # Activation
#         x_after_gelu = gelu(x_after_w1)

#         # Second MLP layer
#         x_after_w2 = stk.ops.dsd(x_after_gelu, mb_model.w2)

#         # Scatter back
#         x_scattered = mb_model._scatter_tokens(
#             x_after_w2, indices, bin_ids, expert_weights_flat, tokens_per_expert, padded_bins
#         )
#         x_scattered = rearrange(x_scattered, '(batch seq) hidden -> batch seq hidden', batch=batch_size, seq=seq_len)


#     # STK forward pass
#     with torch.no_grad():
#         # Sort by expert
#         x_sorted, experts_sorted, weights_sorted, inv_indices = stk_model._sort_by_expert(
#             x_flat, weights_stk, experts_stk
#         )

#         # Pad to blocks
#         x_padded, tokens_per_expert_padded, unpad_indices = stk_model._pad_to_blocks(
#             x_sorted, experts_sorted
#         )
#         total_padded = tokens_per_expert_padded.sum()
#         x_padded = x_padded[:total_padded]

#         # Run STK ops
#         topo = stk_model._make_topology(x_padded, tokens_per_expert_padded)

#         block_sparse = stk.ops.sdd(x_padded, stk_model.w1, topo)

#         # Create new matrix with activated data
#         block_sparse_activated = gelu(block_sparse) 

#         expert_output = stk.ops.dsd(block_sparse_activated, stk_model.w2)

#         # Unpad and weight

#         output_unpadded = torch.index_select(expert_output, 0, unpad_indices)

#         output_weighted = output_unpadded * weights_sorted

#         output = output_weighted[inv_indices]

#         num_tokens = batch_size * seq_len
        
#         original_token_indices = (
#             torch.arange(num_tokens * config.num_experts_per_tok, device=output.device)
#             // config.num_experts_per_tok
#         )
        
#         combined_output = torch.zeros((num_tokens, config.n_embd), 
#                                         dtype=output.dtype, device=output.device)
#         combined_output.scatter_add_(
#             0,
#             original_token_indices.unsqueeze(-1).expand(-1, config.n_embd),
#             output
#         )
#         output = combined_output
#         output = rearrange(output, '(batch seq) hidden -> batch seq hidden', batch=batch_size, seq=seq_len)
    
#     print(f"after w2: {torch.allclose(x_after_w2, expert_output)}")
#     print(x_scattered.shape)
#     print(output.shape)
#     print(torch.allclose(x_scattered,output))

#     return mb_model, stk_model


# def compare_gradients():
#     """Compare gradients between MegaBlocks and STK implementations."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     # Create models
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Synchronize ALL weights before forward pass
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     # Create identical input and target
#     batch_size = 2
#     seq_len = 128  # Smaller for easier debugging
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device, requires_grad=True)
#     target = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    
#     # Clone input for each model to ensure independent computation graphs
#     x_mb = x.clone().detach().requires_grad_(True)
#     x_stk = x.clone().detach().requires_grad_(True)
    
#     # Forward pass
#     out_mb, aux_mb, f_mb = mb_model(x_mb)
#     out_stk, aux_stk, f_stk = stk_model(x_stk)
    
#     print("="*50)
#     print("Forward Pass Comparison")
#     print("="*50)
#     print(f"Output allclose: {torch.allclose(out_mb, out_stk, rtol=1e-5, atol=1e-6)}")
#     print(f"Max output diff: {(out_mb - out_stk).abs().max().item():.6e}")
#     print()
    
#     # Compute identical losses
#     # Simple MSE loss for debugging
#     loss_mb = ((out_mb - target) ** 2).mean()
#     loss_stk = ((out_stk - target) ** 2).mean()
    
#     # Add auxiliary losses (these affect gradients!)
#     total_loss_mb = loss_mb + 0.01 * aux_mb['router_z_loss'] + 0.01 * aux_mb['load_balance_loss']
#     total_loss_stk = loss_stk + 0.01 * aux_stk['router_z_loss'] + 0.01 * aux_stk['load_balance_loss']
    
#     print("Loss Comparison")
#     print("="*50)
#     print(f"Main loss diff: {(loss_mb - loss_stk).abs().item():.6e}")
#     print(f"Total loss diff: {(total_loss_mb - total_loss_stk).abs().item():.6e}")
#     print()
    
#     # Backward pass
#     mb_model.zero_grad()
#     stk_model.zero_grad()
    
#     total_loss_mb.backward()
#     total_loss_stk.backward()
    
#     print("Gradient Comparison")
#     print("="*50)
    
#     # Compare input gradients
#     print(f"Input gradient allclose: {torch.allclose(x_mb.grad, x_stk.grad, rtol=1e-5, atol=1e-6)}")
#     print(f"Input grad max diff: {(x_mb.grad - x_stk.grad).abs().max().item():.6e}")
#     print(f"Input grad mean diff: {(x_mb.grad - x_stk.grad).abs().mean().item():.6e}")
#     print()
    
#     # Compare router gradients
#     router_grad_mb = mb_model.router.weight.grad
#     router_grad_stk = stk_model.router.weight.grad
#     if router_grad_mb is not None and router_grad_stk is not None:
#         print(f"Router gradient allclose: {torch.allclose(router_grad_mb, router_grad_stk, rtol=1e-5, atol=1e-6)}")
#         print(f"Router grad max diff: {(router_grad_mb - router_grad_stk).abs().max().item():.6e}")
#         print(f"Router grad mean diff: {(router_grad_mb - router_grad_stk).abs().mean().item():.6e}")
#         print(f"Router grad norm MB: {router_grad_mb.norm().item():.6f}")
#         print(f"Router grad norm STK: {router_grad_stk.norm().item():.6f}")
#     print()
    
#     # Compare w1 gradients
#     w1_grad_mb = mb_model.w1.grad
#     w1_grad_stk = stk_model.w1.grad
#     if w1_grad_mb is not None and w1_grad_stk is not None:
#         print(f"W1 gradient allclose: {torch.allclose(w1_grad_mb, w1_grad_stk, rtol=1e-5, atol=1e-6)}")
#         print(f"W1 grad max diff: {(w1_grad_mb - w1_grad_stk).abs().max().item():.6e}")
#         print(f"W1 grad mean diff: {(w1_grad_mb - w1_grad_stk).abs().mean().item():.6e}")
#         print(f"W1 grad norm MB: {w1_grad_mb.norm().item():.6f}")
#         print(f"W1 grad norm STK: {w1_grad_stk.norm().item():.6f}")
#     print()
    
#     # Compare w2 gradients  
#     w2_grad_mb = mb_model.w2.grad
#     w2_grad_stk = stk_model.w2.grad
#     if w2_grad_mb is not None and w2_grad_stk is not None:
#         print(f"W2 gradient allclose: {torch.allclose(w2_grad_mb, w2_grad_stk, rtol=1e-5, atol=1e-6)}")
#         print(f"W2 grad max diff: {(w2_grad_mb - w2_grad_stk).abs().max().item():.6e}")
#         print(f"W2 grad mean diff: {(w2_grad_mb - w2_grad_stk).abs().mean().item():.6e}")
#         print(f"W2 grad norm MB: {w2_grad_mb.norm().item():.6f}")
#         print(f"W2 grad norm STK: {w2_grad_stk.norm().item():.6f}")
    
#     return mb_model, stk_model

# def debug_gradient_flow():
#     """More detailed gradient flow analysis."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Synchronize weights
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     # Smaller input for debugging
#     x = torch.randn(1, 8, config.n_embd, device=device, requires_grad=True)
#     target = torch.randn(1, 8, config.n_embd, device=device)
    
#     # Forward with gradient tracking
#     x_mb = x.clone().detach().requires_grad_(True)
#     x_stk = x.clone().detach().requires_grad_(True)
    
#     # Add hooks to track intermediate gradients
#     intermediates_mb = {}
#     intermediates_stk = {}
    
#     def make_hook(name, storage):
#         def hook(grad):
#             storage[name] = grad.clone()
#             return grad
#         return hook
    
#     # Forward pass with hooks
#     out_mb, aux_mb, f_mb = mb_model(x_mb)
#     out_stk, aux_stk, f_stk = stk_model(x_stk)
    
#     # Register hooks on outputs
#     out_mb.register_hook(make_hook('output', intermediates_mb))
#     out_stk.register_hook(make_hook('output', intermediates_stk))
    
#     # Simple loss
#     loss_mb = out_mb.pow(2).mean()
#     loss_stk = out_stk.pow(2).mean()
    
#     # Backward
#     loss_mb.backward()
#     loss_stk.backward()
    
#     print("="*50)
#     print("Gradient Flow Analysis")
#     print("="*50)
    
#     # Check if gradients are flowing
#     print("Gradients exist:")
#     print(f"  MB - router: {mb_model.router.weight.grad is not None}")
#     print(f"  MB - w1: {mb_model.w1.grad is not None}")
#     print(f"  MB - w2: {mb_model.w2.grad is not None}")
#     print(f"  STK - router: {stk_model.router.weight.grad is not None}")
#     print(f"  STK - w1: {stk_model.w1.grad is not None}")
#     print(f"  STK - w2: {stk_model.w2.grad is not None}")
#     print()
    
#     # Check for gradient explosion/vanishing
#     if mb_model.w1.grad is not None and stk_model.w1.grad is not None:
#         print("Gradient magnitudes:")
#         print(f"  MB w1 grad: min={mb_model.w1.grad.min():.6e}, max={mb_model.w1.grad.max():.6e}")
#         print(f"  STK w1 grad: min={stk_model.w1.grad.min():.6e}, max={stk_model.w1.grad.max():.6e}")
#         print(f"  MB w2 grad: min={mb_model.w2.grad.min():.6e}, max={mb_model.w2.grad.max():.6e}")
#         print(f"  STK w2 grad: min={stk_model.w2.grad.min():.6e}, max={stk_model.w2.grad.max():.6e}")
    
#     # Check sparsity patterns (MoE specific issue)
#     if mb_model.w1.grad is not None:
#         mb_w1_zeros = (mb_model.w1.grad == 0).sum().item()
#         mb_w1_total = mb_model.w1.grad.numel()
#         stk_w1_zeros = (stk_model.w1.grad == 0).sum().item()
#         stk_w1_total = stk_model.w1.grad.numel()
        
#         print(f"\nGradient sparsity:")
#         print(f"  MB w1: {mb_w1_zeros}/{mb_w1_total} zeros ({100*mb_w1_zeros/mb_w1_total:.1f}%)")
#         print(f"  STK w1: {stk_w1_zeros}/{stk_w1_total} zeros ({100*stk_w1_zeros/stk_w1_total:.1f}%)")


# if __name__ == "__main__":
#     # main()
#     # compare_forward_passes()
#     compare_gradients()
#     print("\n" + "="*50 + "\n")
#     debug_gradient_flow()


# def compare_training_dynamics():
#     """Compare how the models diverge during actual training."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     # Create two pairs of models - one synced, one trained
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Sync initial weights completely
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     # Store initial weights for comparison
#     init_router = mb_model.router.weight.clone()
#     init_w1 = mb_model.w1.clone()
#     init_w2 = mb_model.w2.clone()
    
#     # Training setup
#     lr = 0.01  # High LR to see differences quickly
#     steps = 10000
#     batch_size = 2
#     seq_len = 128
    
#     # Track metrics
#     mb_losses = []
#     stk_losses = []
#     weight_diffs = []
#     grad_norm_ratios = []
    
#     print("="*50)
#     print("Training Dynamics Comparison")
#     print("="*50)
    
#     for step in range(steps):
#         # Same input/target for both
#         x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
#         target = torch.randn(batch_size, seq_len, config.n_embd, device=device)
        
#         # Forward pass
#         out_mb, aux_mb, _ = mb_model(x)
#         out_stk, aux_stk, _ = stk_model(x)
        
#         # Compute losses
#         loss_mb = ((out_mb - target) ** 2).mean()
#         loss_stk = ((out_stk - target) ** 2).mean()
        
#         # Add auxiliary losses (important for MoE training!)
#         total_loss_mb = loss_mb + 0.01 * aux_mb['router_z_loss'] + 0.01 * aux_mb['load_balance_loss']
#         total_loss_stk = loss_stk + 0.01 * aux_stk['router_z_loss'] + 0.01 * aux_stk['load_balance_loss']
        
#         mb_losses.append(loss_mb.item())
#         stk_losses.append(loss_stk.item())
        
#         # Backward
#         mb_model.zero_grad()
#         stk_model.zero_grad()
        
#         total_loss_mb.backward()
#         total_loss_stk.backward()
        
#         # Track gradient norms
#         mb_grad_norm = mb_model.w1.grad.norm().item()
#         stk_grad_norm = stk_model.w1.grad.norm().item()
#         grad_norm_ratios.append(stk_grad_norm / (mb_grad_norm + 1e-8))
        
#         # Manual SGD update to ensure identical optimization
#         with torch.no_grad():
#             # MegaBlocks update
#             mb_model.router.weight -= lr * mb_model.router.weight.grad
#             mb_model.w1 -= lr * mb_model.w1.grad
#             mb_model.w2 -= lr * mb_model.w2.grad
            
#             # STK update
#             stk_model.router.weight -= lr * stk_model.router.weight.grad
#             stk_model.w1 -= lr * stk_model.w1.grad
#             stk_model.w2 -= lr * stk_model.w2.grad
        
#         # Track weight divergence
#         w1_diff = (mb_model.w1 - stk_model.w1).abs().max().item()
#         weight_diffs.append(w1_diff)
        
#         if step % 2 == 0:
#             print(f"Step {step:2d}: Loss MB={loss_mb:.6f}, STK={loss_stk:.6f}, "
#                   f"W1 diff={w1_diff:.6e}, Grad ratio={grad_norm_ratios[-1]:.4f}")
    
#     print("\n" + "="*50)
#     print("Summary Statistics")
#     print("="*50)
    
#     # How much did weights change from initialization?
#     mb_w1_change = (mb_model.w1 - init_w1).abs().mean().item()
#     stk_w1_change = (stk_model.w1 - init_w1).abs().mean().item()
    
#     print(f"Average loss - MB: {np.mean(mb_losses):.6f}, STK: {np.mean(stk_losses):.6f}")
#     print(f"Final weight divergence: {weight_diffs[-1]:.6e}")
#     print(f"W1 change from init - MB: {mb_w1_change:.6e}, STK: {stk_w1_change:.6e}")
#     print(f"Average gradient norm ratio (STK/MB): {np.mean(grad_norm_ratios):.4f}")
    
#     # Plot if matplotlib available
#     try:
#         import matplotlib.pyplot as plt
#         fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
#         axes[0].plot(mb_losses, label='MegaBlocks', marker='o')
#         axes[0].plot(stk_losses, label='STK', marker='s')
#         axes[0].set_xlabel('Step')
#         axes[0].set_ylabel('Loss')
#         axes[0].set_title('Training Loss')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
        
#         axes[1].plot(weight_diffs, marker='o', color='red')
#         axes[1].set_xlabel('Step')
#         axes[1].set_ylabel('Max Weight Difference')
#         axes[1].set_title('Weight Divergence')
#         axes[1].set_yscale('log')
#         axes[1].grid(True, alpha=0.3)
        
#         axes[2].plot(grad_norm_ratios, marker='o', color='green')
#         axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
#         axes[2].set_xlabel('Step')
#         axes[2].set_ylabel('Gradient Norm Ratio (STK/MB)')
#         axes[2].set_title('Gradient Magnitude Comparison')
#         axes[2].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.show()
#     except ImportError:
#         pass
    
#     return mb_model, stk_model

# def check_numerical_stability():
#     """Check for numerical stability issues in the operations."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Sync weights
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     print("="*50)
#     print("Numerical Stability Check")
#     print("="*50)
    
#     # Test with different input magnitudes
#     for scale in [1e-3, 1.0, 1e3]:
#         x = torch.randn(2, 64, config.n_embd, device=device) * scale
        
#         out_mb, _, _ = mb_model(x)
#         out_stk, _, _ = stk_model(x)
        
#         diff = (out_mb - out_stk).abs().max().item()
#         relative_diff = diff / (out_mb.abs().mean().item() + 1e-8)
        
#         print(f"Input scale {scale:7.1e}: Max diff={diff:.6e}, Relative diff={relative_diff:.6e}")
    
#     # Check gradient accumulation with mixed precision
#     x = torch.randn(2, 64, config.n_embd, device=device)
    
#     # Multiple backward passes to check accumulation
#     for i in range(3):
#         out_mb, aux_mb, _ = mb_model(x)
#         out_stk, aux_stk, _ = stk_model(x)
        
#         loss_mb = out_mb.pow(2).mean()
#         loss_stk = out_stk.pow(2).mean()
        
#         loss_mb.backward()
#         loss_stk.backward()
        
#         if i == 0:
#             print(f"\nGradient accumulation (step {i+1}):")
        
#         w1_grad_diff = (mb_model.w1.grad - stk_model.w1.grad).abs().max().item()
#         print(f"  Step {i+1} - W1 grad diff: {w1_grad_diff:.6e}")

# if __name__ == "__main__":
#     # First check the training dynamics
#     compare_training_dynamics()
    
#     print("\n" + "="*50 + "\n")
    
#     # Then check numerical stability
#     check_numerical_stability()

# def trace_gradient_flow_fixed():
#     """Properly trace gradient flow through both implementations."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Sync weights
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     batch_size = 2
#     seq_len = 8
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    
#     # Hook into the actual modules
#     mb_intermediates = {}
#     stk_intermediates = {}
    
#     def make_forward_hook(name, storage):
#         def hook(module, input, output):
#             storage[f'{name}_out'] = output.detach().clone()
#             # Register backward hook on the output
#             if output.requires_grad:
#                 handle = output.register_hook(lambda grad: storage.update({f'{name}_grad': grad.clone()}))
#         return hook
    
#     # Register hooks on the router modules themselves
#     mb_handle = mb_model.router.register_forward_hook(make_forward_hook('router', mb_intermediates))
#     stk_handle = stk_model.router.register_forward_hook(make_forward_hook('router', stk_intermediates))
    
#     # Forward pass
#     out_mb, aux_mb, _ = mb_model(x)
#     out_stk, aux_stk, _ = stk_model(x)
    
#     # Simple loss - use sum to get full gradients
#     loss_mb = out_mb.sum()
#     loss_stk = out_stk.sum()
    
#     # Backward
#     loss_mb.backward()
#     loss_stk.backward()
    
#     # Clean up hooks
#     mb_handle.remove()
#     stk_handle.remove()
    
#     print("="*50)
#     print("Fixed Gradient Flow Analysis")
#     print("="*50)
    
#     if 'router_grad' in mb_intermediates and 'router_grad' in stk_intermediates:
#         mb_router_grad = mb_intermediates['router_grad']
#         stk_router_grad = stk_intermediates['router_grad']
        
#         print(f"Router output gradient norms:")
#         print(f"  MegaBlocks: {mb_router_grad.norm():.6f}")
#         print(f"  STK: {stk_router_grad.norm():.6f}")
#         print(f"  Ratio (STK/MB): {stk_router_grad.norm() / mb_router_grad.norm():.4f}")
    
#     print(f"\nWeight gradient statistics:")
#     print(f"W1 gradients:")
#     print(f"  MB norm: {mb_model.w1.grad.norm():.6f}")
#     print(f"  STK norm: {stk_model.w1.grad.norm():.6f}")
#     print(f"  Ratio: {stk_model.w1.grad.norm() / mb_model.w1.grad.norm():.4f}")
    
#     print(f"\nW2 gradients:")
#     print(f"  MB norm: {mb_model.w2.grad.norm():.6f}")
#     print(f"  STK norm: {stk_model.w2.grad.norm():.6f}")
#     print(f"  Ratio: {stk_model.w2.grad.norm() / mb_model.w2.grad.norm():.4f}")
    
#     print(f"\nRouter weight gradients:")
#     print(f"  MB norm: {mb_model.router.weight.grad.norm():.6f}")
#     print(f"  STK norm: {stk_model.router.weight.grad.norm():.6f}")
#     print(f"  Ratio: {stk_model.router.weight.grad.norm() / mb_model.router.weight.grad.norm():.4f}")
    
#     return mb_model, stk_model

# def analyze_scatter_operations():
#     """Directly test the scatter operations used by both models."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Sync weights
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     # Small controlled input
#     batch_size = 1
#     seq_len = 4
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device, requires_grad=True)
#     x_mb = x.clone().requires_grad_(True)
#     x_stk = x.clone().requires_grad_(True)
    
#     # Run through both models and check intermediate operations
#     x_flat_mb = x_mb.reshape(-1, config.n_embd)
#     x_flat_stk = x_stk.reshape(-1, config.n_embd)
    
#     # Router step
#     router_logits_mb = mb_model.router(x_flat_mb)
#     router_logits_stk = stk_model.router(x_flat_stk)
    
#     # Softmax and top-k (this is where things might diverge)
#     probs_mb = F.softmax(router_logits_mb, dim=-1, dtype=torch.float)
#     probs_stk = F.softmax(router_logits_stk, dim=-1, dtype=torch.float)
    
#     weights_mb, experts_mb = torch.topk(probs_mb, config.num_experts_per_tok)
#     weights_stk, experts_stk = torch.topk(probs_stk, config.num_experts_per_tok)
    
#     # Normalize
#     if config.norm_topk_prob:
#         weights_mb_norm = weights_mb / weights_mb.sum(dim=-1, keepdim=True)
#         weights_stk_norm = weights_stk / weights_stk.sum(dim=-1, keepdim=True)
    
#     # Test gradient flow through normalization
#     loss_mb = weights_mb_norm.sum()
#     loss_stk = weights_stk_norm.sum()
    
#     grad_mb = torch.autograd.grad(loss_mb, router_logits_mb, retain_graph=True)[0]
#     grad_stk = torch.autograd.grad(loss_stk, router_logits_stk, retain_graph=True)[0]
    
#     print("="*50)
#     print("Router → Weight Gradient Flow")
#     print("="*50)
#     print(f"Gradient norm through softmax+topk+normalize:")
#     print(f"  MegaBlocks: {grad_mb.norm():.6f}")
#     print(f"  STK: {grad_stk.norm():.6f}")
#     print(f"  Ratio (STK/MB): {grad_stk.norm() / grad_mb.norm():.4f}")
    
#     # Check if gradients are exactly equal
#     print(f"  Exactly equal: {torch.allclose(grad_mb, grad_stk)}")
#     print(f"  Max difference: {(grad_mb - grad_stk).abs().max():.6e}")

# def check_expert_weight_application():
#     """Check how expert weights are applied in scatter operations."""
#     torch.manual_seed(42)
    
#     # Simulate the final scatter with weights
#     num_tokens = 16
#     hidden_dim = 8
#     num_experts_per_tok = 2
    
#     # Create some dummy expert outputs
#     expert_outputs = torch.randn(num_tokens * num_experts_per_tok, hidden_dim, 
#                                  device=device, requires_grad=True)
    
#     # Create weights (these would come from router)
#     weights = torch.rand(num_tokens * num_experts_per_tok, 1, device=device)
#     weights = weights / weights.sum()  # Normalize for comparison
    
#     # Method 1: Multiply then scatter (MegaBlocks style)
#     weighted_outputs_mb = expert_outputs * weights
#     final_mb = torch.zeros(num_tokens, hidden_dim, device=device)
#     indices = torch.arange(num_tokens, device=device).repeat_interleave(num_experts_per_tok)
#     final_mb.scatter_add_(0, indices.unsqueeze(-1).expand(-1, hidden_dim), weighted_outputs_mb)
    
#     # Method 2: Scatter then weight (potential STK difference)
#     final_stk = torch.zeros(num_tokens, hidden_dim, device=device)
#     # STK might accumulate differently
#     for i in range(num_tokens):
#         token_experts = expert_outputs[i*num_experts_per_tok:(i+1)*num_experts_per_tok]
#         token_weights = weights[i*num_experts_per_tok:(i+1)*num_experts_per_tok]
#         final_stk[i] = (token_experts * token_weights).sum(dim=0)
    
#     # Check gradient flow
#     loss_mb = final_mb.sum()
#     loss_stk = final_stk.sum()
    
#     grad_mb = torch.autograd.grad(loss_mb, expert_outputs, retain_graph=True)[0]
#     grad_stk = torch.autograd.grad(loss_stk, expert_outputs, retain_graph=True)[0]
    
#     print("="*50)
#     print("Weight Application in Scatter")
#     print("="*50)
#     print(f"Gradient norms:")
#     print(f"  Method 1 (MB): {grad_mb.norm():.6f}")
#     print(f"  Method 2 (STK): {grad_stk.norm():.6f}")
#     print(f"  Ratio: {grad_stk.norm() / grad_mb.norm():.4f}")
    
#     # Check where they differ
#     diff_mask = ~torch.isclose(grad_mb, grad_stk)
#     if diff_mask.any():
#         print(f"  Gradients differ at {diff_mask.sum().item()} positions")
#         print(f"  Max difference: {(grad_mb - grad_stk).abs().max():.6e}")

# def trace_sparse_ops_gradients():
#     """Trace gradients through the actual sparse operations."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Sync weights
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     # Very small input to trace exactly what's happening
#     batch_size = 1
#     seq_len = 4  
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
#     x_flat = x.reshape(-1, config.n_embd)
    
#     # Get routing (should be identical)
#     router_logits = mb_model.router(x_flat)
#     probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
#     weights, experts = torch.topk(probs, config.num_experts_per_tok)
#     weights = weights / weights.sum(dim=-1, keepdim=True)
#     weights = weights.to(x.dtype)
    
#     print("="*50)
#     print("Sparse Operation Gradient Analysis")
#     print("="*50)
    
#     # MegaBlocks path
#     expert_weights_flat = weights.reshape(-1)
#     selected_experts_flat = experts.reshape(-1)
    
#     bin_ids, indices, tokens_per_expert = mb_model._sort_tokens_by_expert(selected_experts_flat)
#     padded_bins, topology = mb_model._create_topology(x_flat, tokens_per_expert)
#     x_permuted_mb = mb_model._gather_tokens(x_flat, indices, bin_ids, tokens_per_expert, padded_bins)
    
#     # Pad if needed
#     if x_permuted_mb.shape[0] != topology.shape[0]:
#         padding = torch.zeros((topology.shape[0] - x_permuted_mb.shape[0], x_permuted_mb.shape[1]),
#                              dtype=x_permuted_mb.dtype, device=x_permuted_mb.device)
#         x_permuted_mb = torch.cat([x_permuted_mb, padding], dim=0)
    
#     num_real_tokens = min(x_permuted_mb.shape[0], topology.shape[0])
#     num_padded = topology.shape[0] - num_real_tokens
#     print(f"MegaBlocks: {num_real_tokens} real tokens, {num_padded} padded tokens")
    
#     # First linear through sparse op
#     x_permuted_mb.requires_grad_(True)
#     mb_model.w1.requires_grad_(True)
    
#     hidden_mb = stk.ops.sdd(x_permuted_mb, mb_model.w1, topology)
#     hidden_mb_activated = gelu(hidden_mb)
    
#     # Now we need to handle the Matrix type - gelu returns a Matrix
#     # We'll use dsd to multiply by an identity-like matrix to get a tensor output
#     # Or we can directly test the sparse operations before gelu
    
#     # Let's test without gelu first to isolate the sparse op
#     output_mb = stk.ops.dsd(hidden_mb, mb_model.w2)  # This gives us a tensor
    
#     # Create loss and get gradients
#     loss_mb = output_mb.sum()
    
#     # Get gradients
#     input_grad_mb = torch.autograd.grad(loss_mb, x_permuted_mb, retain_graph=True)[0]
#     w1_grad_mb = torch.autograd.grad(loss_mb, mb_model.w1, retain_graph=True)[0]
#     w2_grad_mb = torch.autograd.grad(loss_mb, mb_model.w2, retain_graph=True)[0]
    
#     print(f"\nMegaBlocks gradient norms (full path):")
#     print(f"  Input gradient: {input_grad_mb.norm():.6f}")
#     print(f"  W1 gradient: {w1_grad_mb.norm():.6f}")
#     print(f"  W2 gradient: {w2_grad_mb.norm():.6f}")
#     print(f"  Zero gradients in input: {(input_grad_mb == 0).sum().item()} / {input_grad_mb.numel()}")
    
#     # STK path
#     x_sorted, experts_sorted, weights_sorted, inv_indices = stk_model._sort_by_expert(
#         x_flat, weights, experts
#     )
#     x_padded_stk, tokens_per_expert_padded, unpad_indices = stk_model._pad_to_blocks(x_sorted, experts_sorted)
#     total_padded_tokens = tokens_per_expert_padded.sum()
#     x_padded_stk = x_padded_stk[:total_padded_tokens]
    
#     print(f"\nSTK: {unpad_indices.shape[0]} real tokens, {total_padded_tokens - unpad_indices.shape[0]} padded tokens")
    
#     # STK sparse ops
#     x_padded_stk.requires_grad_(True)
#     stk_model.w1.requires_grad_(True)
    
#     topo_stk = stk_model._make_topology(x_padded_stk, tokens_per_expert_padded)
#     hidden_stk = stk.ops.sdd(x_padded_stk, stk_model.w1, topo_stk)
#     hidden_stk_activated = gelu(hidden_stk)
#     output_stk = stk.ops.dsd(hidden_stk_activated, stk_model.w2)
    
#     # Get gradient
#     loss_stk = output_stk.sum()
    
#     input_grad_stk = torch.autograd.grad(loss_stk, x_padded_stk, retain_graph=True)[0]
#     w1_grad_stk = torch.autograd.grad(loss_stk, stk_model.w1, retain_graph=True)[0]
#     w2_grad_stk = torch.autograd.grad(loss_stk, stk_model.w2, retain_graph=True)[0]
    
#     print(f"\nSTK gradient norms (full path):")
#     print(f"  Input gradient: {input_grad_stk.norm():.6f}")
#     print(f"  W1 gradient: {w1_grad_stk.norm():.6f}")
#     print(f"  W2 gradient: {w2_grad_stk.norm():.6f}")
#     print(f"  Zero gradients in input: {(input_grad_stk == 0).sum().item()} / {input_grad_stk.numel()}")
    
#     print(f"\nGradient ratio (STK/MB):")
#     print(f"  Input gradients: {input_grad_stk.norm() / input_grad_mb.norm():.4f}")
#     print(f"  W1 gradients: {w1_grad_stk.norm() / w1_grad_mb.norm():.4f}")
#     print(f"  W2 gradients: {w2_grad_stk.norm() / w2_grad_mb.norm():.4f}")
    
#     # Check gradient distribution
#     print(f"\nGradient statistics:")
#     print(f"MegaBlocks W1 grad: mean={w1_grad_mb.mean():.6e}, std={w1_grad_mb.std():.6e}")
#     print(f"STK W1 grad: mean={w1_grad_stk.mean():.6e}, std={w1_grad_stk.std():.6e}")
    
#     # Check if specific experts have different gradients
#     # W1 shape is (n_embd, d_ffn * num_experts)
#     experts_dim = config.d_ffn
#     for expert_id in range(min(3, config.num_experts)):  # Check first 3 experts
#         start_idx = expert_id * experts_dim
#         end_idx = (expert_id + 1) * experts_dim
#         mb_expert_grad = w1_grad_mb[:, start_idx:end_idx]
#         stk_expert_grad = w1_grad_stk[:, start_idx:end_idx]
#         print(f"\nExpert {expert_id} W1 gradients:")
#         print(f"  MB norm: {mb_expert_grad.norm():.6f}")
#         print(f"  STK norm: {stk_expert_grad.norm():.6f}")
#         print(f"  Ratio: {stk_expert_grad.norm() / (mb_expert_grad.norm() + 1e-8):.4f}")


# def check_topology_differences():
#     """Compare the topologies created by both models."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     batch_size = 1
#     seq_len = 4
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
#     x_flat = x.reshape(-1, config.n_embd)
    
#     # Get routing
#     router_logits = mb_model.router(x_flat)
#     probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
#     weights, experts = torch.topk(probs, config.num_experts_per_tok)
    
#     # MegaBlocks topology
#     selected_experts_flat = experts.reshape(-1)
#     _, _, tokens_per_expert = mb_model._sort_tokens_by_expert(selected_experts_flat)
#     padded_bins, topology_mb = mb_model._create_topology(x_flat, tokens_per_expert)
    
#     # STK topology  
#     x_sorted, experts_sorted, _, _ = stk_model._sort_by_expert(x_flat, weights, experts)
#     _, tokens_per_expert_padded, _ = stk_model._pad_to_blocks(x_sorted, experts_sorted)
#     x_padded_stk = torch.zeros(tokens_per_expert_padded.sum(), config.n_embd, device=device)
#     topology_stk = stk_model._make_topology(x_padded_stk, tokens_per_expert_padded)
    
#     print("="*50)
#     print("Topology Comparison")
#     print("="*50)
#     print(f"MegaBlocks topology shape: {topology_mb.shape}")
#     print(f"STK topology shape: {topology_stk.shape}")
#     print(f"MegaBlocks nnz: {topology_mb.data.shape[0]}")
#     print(f"STK nnz: {topology_stk.data.shape[0]}")
    
#     # The topology affects how gradients flow!
#     # Different blocking might lead to different gradient accumulation

# if __name__ == "__main__":
#     trace_sparse_ops_gradients()
#     print("\n")
#     check_topology_differences()


# def debug_token_counts():
#     """Figure out why MegaBlocks is processing 640 tokens."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
    
#     batch_size = 64
#     seq_len = 256
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    
#     print(f"Input: batch_size={batch_size}, seq_len={seq_len}")
#     print(f"Expected token-expert pairs: {seq_len * config.num_experts_per_tok}")
    
#     x_flat = x.reshape(-1, config.n_embd)
#     print(f"x_flat shape: {x_flat.shape}")
    
#     # Get routing
#     router_logits = mb_model.router(x_flat)
#     probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
#     weights, experts = torch.topk(probs, config.num_experts_per_tok)
    
#     selected_experts_flat = experts.reshape(-1)
#     print(f"selected_experts_flat shape: {selected_experts_flat.shape}")
    
#     # This is where things might go wrong
#     bin_ids, indices, tokens_per_expert = mb_model._sort_tokens_by_expert(selected_experts_flat)
#     print(f"tokens_per_expert: {tokens_per_expert}")
#     print(f"tokens_per_expert sum: {tokens_per_expert.sum()}")
    
#     padded_bins, topology = mb_model._create_topology(x_flat, tokens_per_expert)
#     print(f"padded_bins: {padded_bins}")
#     print(f"topology shape: {topology.shape}")
    
#     x_permuted = mb_model._gather_tokens(x_flat, indices, bin_ids, tokens_per_expert, padded_bins)
#     print(f"x_permuted shape after gather: {x_permuted.shape}")

# if __name__ == "__main__":
#     debug_token_counts()


# def test_realistic_batch():
#     """Test with realistic batch size to see true gradient differences."""
#     torch.manual_seed(42)
#     config = TestConfig()
    
#     mb_model = MoeMLPMegaBlocks(config).to(device)
#     stk_model = MoeMLPSTK(config).to(device)
    
#     # Sync all weights
#     with torch.no_grad():
#         stk_model.router.weight.copy_(mb_model.router.weight)
#         stk_model.w1.copy_(mb_model.w1)
#         stk_model.w2.copy_(mb_model.w2)
    
#     # Realistic batch size
#     batch_size = 64
#     seq_len = 256
#     x = torch.randn(batch_size, seq_len, config.n_embd, device=device)
    
#     print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
#     print(f"Total tokens: {batch_size * seq_len}")
#     print()
    
#     # Forward pass through both models
#     out_mb, aux_mb, f_mb = mb_model(x)
#     out_stk, aux_stk, f_stk = stk_model(x)
    
#     # Check forward pass match
#     print("Forward pass comparison:")
#     print(f"  Outputs match: {torch.allclose(out_mb, out_stk, rtol=1e-4, atol=1e-5)}")
#     print(f"  Max difference: {(out_mb - out_stk).abs().max().item():.6e}")
#     print()
    
#     # Create identical loss
#     target = torch.randn_like(out_mb)
#     loss_mb = ((out_mb - target) ** 2).mean()
#     loss_stk = ((out_stk - target) ** 2).mean()
    
#     # Backward pass
#     mb_model.zero_grad()
#     stk_model.zero_grad()
    
#     loss_mb.backward()
#     loss_stk.backward()
    
#     # Compare gradients
#     print("Gradient comparison:")
#     print(f"Router gradients:")
#     print(f"  MB norm: {mb_model.router.weight.grad.norm():.6f}")
#     print(f"  STK norm: {stk_model.router.weight.grad.norm():.6f}")
#     print(f"  Ratio (STK/MB): {stk_model.router.weight.grad.norm() / mb_model.router.weight.grad.norm():.4f}")
#     print()
    
#     print(f"W1 gradients:")
#     print(f"  MB norm: {mb_model.w1.grad.norm():.6f}")
#     print(f"  STK norm: {stk_model.w1.grad.norm():.6f}")
#     print(f"  Ratio (STK/MB): {stk_model.w1.grad.norm() / mb_model.w1.grad.norm():.4f}")
#     print()
    
#     print(f"W2 gradients:")
#     print(f"  MB norm: {mb_model.w2.grad.norm():.6f}")
#     print(f"  STK norm: {stk_model.w2.grad.norm():.6f}")
#     print(f"  Ratio (STK/MB): {stk_model.w2.grad.norm() / mb_model.w2.grad.norm():.4f}")
    
#     # Check gradient sparsity
#     print(f"\nGradient sparsity:")
#     mb_w1_zeros = (mb_model.w1.grad == 0).sum().item() 
#     stk_w1_zeros = (stk_model.w1.grad == 0).sum().item()
#     print(f"  MB W1 zeros: {mb_w1_zeros} / {mb_model.w1.grad.numel()} ({100*mb_w1_zeros/mb_model.w1.grad.numel():.2f}%)")
#     print(f"  STK W1 zeros: {stk_w1_zeros} / {stk_model.w1.grad.numel()} ({100*stk_w1_zeros/stk_model.w1.grad.numel():.2f}%)")
    
#     return mb_model, stk_model

# if __name__ == "__main__":
#     test_realistic_batch()



