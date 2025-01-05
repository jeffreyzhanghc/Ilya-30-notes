---
layout: post
title: "DeepSeekv3 Paper Deep Dive"
date: 2025-1-2
author: Jeffrey Zhang
math: true
permalink: /posts/deepseekv3.html
---
## Table of Contents

1. [Overview](#overview)
2. [Rotary Position Embeddings (RoPE)](#2-rotary-position-embeddings-rope)
3. [Multi-Layer Perceptron](#3-multi-layer-perceptron)
4. [Attention Implementation](#4-attention-implementation)
5. [DecodeLayer](#5-decodelayer)
6. [Putting Together](#6-putting-together)
7. [Extra: Implementation Details](#extra-then-how-does-the-generate-function-in-huggingface-work)

## Overview
This code represents the implementation of LLaMA (Large Language Model Meta AI) in the Hugging Face Transformers library. The implementation includes several key components working together to create an efficient and scalable language model.




### Brief Overview about DeepSeek v2
#### Multi-head Latent Attention
![alt text](image.png)
DeepSeek leverages the power of multi-head attention instead of traditional multi-head attention to reduce the heavy KV cache that limits the maximum batch size and sequence length. Since the normal multi-head attention includes the following elements:

$\mathbf{q}_t$, $\mathbf{k}_t$, $\mathbf{v}_t$ is first computed by multiplying the corresponding weight matrix with inputs, and then they  will be sliced into $n_h$ heads for the multi-head attention computation:

$$[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_h}] = \mathbf{q}_t,\tag{4}$$

$$[\mathbf{k}_{t,1};\mathbf{k}_{t,2};...;\mathbf{k}_{t,n_h}] = \mathbf{k}_t,\tag{5}$$

$$[\mathbf{v}_{t,1};\mathbf{v}_{t,2};...;\mathbf{v}_{t,n_h}] = \mathbf{v}_t,\tag{6}$$

$$\mathbf{o}_{t,i} = \sum_{j=1}^t \text{Softmax}_j(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h}})\mathbf{v}_{j,i},\tag{7}$$

$$\mathbf{u}_t = W^O[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_h}],\tag{8}$$

We need to cache $$2n_hd_hL$$ because For each token, we need to cache both keys (k) and values (v) - this explains the factor of 2
Each key and value has dimension d_h (hidden dimension per head). There are n_h attention heads ,and this needs to be done for sequence length L tokens.

Instead, what DeepSeek V2 did is reducing KV cache through low-rank joint compression:

$$\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t,\tag{9}$$

$$\mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV},\tag{10}$$

$$\mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV},\tag{11}$$

where:

- $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ is the compressed shared latent vector for both keys and values
- $d_c$ is the compression dimension, which is much smaller than $d_hn_h$
- $W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the down-projection matrix
- $W^{UK}, W^{UV} \in \mathbb{R}^{d_hn_h \times d_c}$ are up-projection matrices for keys and values

Now we only need to cache $d_cl$ elements, where l is the number of layers.

#### Decoupled RoPE
If you are not familiar with RoPE, I also have brief intro written [here](https://jeffreyzhanghc.github.io/Ilya-30-notes/posts/llama-implementation.html#2-rotary-position-embeddings-rope).

We wanted to incorporate both Rotary Position Embedding (RoPE) and low-rank KV compression, but these two techniques were fundamentally incompatible. Here's why:

RoPE applies position-sensitive transformations to both keys and queries. When using RoPE with compressed keys (k^C), the up-projection matrix W^UK becomes entangled with position-specific RoPE matrices. Due to the non-commutative nature of matrix multiplication, we can no longer absorb W^UK into W^Q during inference - a key optimization in the original design.

To resolve this conflict, a "decoupled RoPE strategy" was developed. The core idea is elegant: we separate the positional encoding concerns from the content processing by introducing:

1. Additional multi-head queries (q^R) and a shared key (k^R) specifically for handling positional information
2. These operate in their own dimension space (d^R_h)
3. The regular content-based processing continues through the compressed pathway

The computation flow becomes:
- Position-aware components are processed through RoPE(W^QR c^Q) and RoPE(W^KR h_t)
- Content information flows through the compressed pathway
- These are concatenated before the attention computation
- Finally, everything comes together in a unified attention mechanism that considers both content and position










