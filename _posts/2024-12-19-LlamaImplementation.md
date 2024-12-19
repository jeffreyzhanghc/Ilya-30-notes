---
layout: post
title: "Annotated Transformer and Llama Implementation"
date: 2024-12-19
author: Jeffrey Zhang
math: true
permalink: /posts/llama-implementation.html
---

# LLaMA Implementation Analysis

## Overview
This code represents the implementation of LLaMA (Large Language Model Meta AI) in the Hugging Face Transformers library. The implementation includes several key components working together to create an efficient and scalable language model.

## Key Components

### 1. LlamaRMSNorm
RMS (Root Mean Square) Normalization layer, which is more efficient than traditional layer normalization.
Below are some key take aways:
- LayerNorm is used to address internal covariate shift issue. When deep neural networks process data through multiple layers, the distribution of activations can shift dramatically (internal covariate shift). As data flows through the network, the mean and variance of activations can grow or shrink unpredictably, which lead to unstable training.
- the decoupling from batch-based samples endows LayerNorm with the superiority over batch normalization in handling    variable-length sequences using RNNs.
- RMS norm only focus on re-scaling the invariances, and not re-center it as LayerNorm do, trade weight matrix - recentering for data set re-scaling.
The source paper explaining the RMS can be found [here](https://arxiv.org/pdf/1910.07467).

The formula of RMS Norm is below:
$$\bar{a_i} = \frac{a_i}{\text{RMS}(\mathbf{a})}g_i, \quad \text{where} \quad \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^n a_i^2}.$$

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

Key features:
- Uses parameter-efficient scaling factor
- Maintains numerical stability with epsilon
- Handles dtype conversion automatically
- More computationally efficient than LayerNorm

### 2. Rotary Position Embeddings (RoPE)
Implements rotary positional embeddings for sequence position encoding.

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # Configuration setup
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
```

Core RoPE functionality:
```python
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE computation
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### 3. Attention Mechanism
Multi-head attention implementation with several optimizations:

```python
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
```

Key attention computation:
```python
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, ...):
        bsz, q_len, _ = hidden_states.size()
        
        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape and apply rotary embeddings
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
```

### 4. MLP Block
Feed-forward network implementation with SwiGLU activation:

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            # Tensor parallelism implementation
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            
            # Parallel computation
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) 
                                 for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) 
                               for i in range(self.config.pretraining_tp)], dim=-1)
        else:
            # Standard computation
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
```

## Implementation Details

### 1. Flash Attention Implementation
```python
class LlamaFlashAttention2(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
```

### 2. Scaled Dot Product Attention
```python
class LlamaSdpaAttention(LlamaAttention):
    def forward(self, hidden_states, attention_mask=None, ...):
        # SDPA implementation
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
```

## Best Practices and Usage Guidelines

### Memory Optimization
1. Use Flash Attention when possible:
```python
config._attn_implementation = "flash_attention_2"
```

2. Enable gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```

### Performance Optimization
1. Configure tensor parallelism:
```python
config.pretraining_tp = num_gpus
```

2. Use efficient attention variants:
```python
# Choose between implementations
attention_implementation = "sdpa"  # or "flash_attention_2" or "eager"
```

## Technical Requirements
```python
requirements = {
    "pytorch": ">=2.0.0",
    "transformers": ">=4.36.0",
    "flash-attn": ">=2.0.0",  # Optional for Flash Attention
    "cuda": ">=11.6"  # For GPU acceleration
}
```

## Conclusion
The LLaMA implementation showcases several advanced features and optimizations:
- Efficient attention mechanisms with multiple implementations
- Advanced positional encoding with RoPE
- Memory-efficient normalization
- Support for tensor parallelism
- Comprehensive caching mechanisms

The code is designed for both research and production use, with careful attention to performance and memory efficiency while maintaining flexibility and extensibility.