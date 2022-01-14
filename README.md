# Transformer
Transformer Model

# Model Architecture

---

### Encoder and Decoder

> **Encoder**
The encoder is composed of a stack of $N=6$ identical layers
Each layer has two sub-layers.
> 
> 1. 1st sub-layer : a multi-head self-attention mechanism
> 2. 2nd sub-layer : a simple, position-wise fully connected feed-forward network
> 3. a residual connection around each of the two sub-layers, followed by layer normalization
> 4. To facilitate these residual connections, all sub-layers in the model,  as well as the embedding layers, produce outputs of dimension of $d_{model}=512$

> Decoder
The decoder is also composed of stack of $N=6$ identical layers
Each layer has two sub-layers + `1 sub-layers`
> 
> - additional sub-layer : a masked multi-head self-attention to prevent positions from attending to subsequent positions

![Screen Shot 2022-01-02 at 22.56.39.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/393bf4e4-f903-4a4f-9044-a6513ae947a6/Screen_Shot_2022-01-02_at_22.56.39.png)

### Attention

> query
> 
> 
> key - value
> 

### Scaled Dot-Product Attention

> 
> 
> 
> ![Screen Shot 2022-01-02 at 23.35.08.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/279e3c08-060c-47dc-8da9-14b8a9d19220/Screen_Shot_2022-01-02_at_23.35.08.png)
> 
> $Attention(Q, K, V) = softmax(\frac{\large QK^T}{\large\sqrt{d_k}})V$
> 

### Multi-Head Attention

> 
> 
> 
> ![Screen Shot 2022-01-02 at 23.44.57.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ee818d0b-7e82-49bf-8672-1edffd1765a2/Screen_Shot_2022-01-02_at_23.44.57.png)
> 
> $MultiHead(Q,K,V)=Concat(head_1, ..., head_h)W^o$
> 
> where  $head_i=Attention(QW^Q_i, KW^K_i, VW^V_i)$
> 

$\large W^Q_i ∈ \R^{d_{model}×d_k} , 
W^K_i ∈ \R^{d_{model}×d_k} , 
W^V_i ∈ \R^{d_{model}×d_v} ,
W^O ∈ \R^{hd_v×d_{model}}$

In this work, they employ $h=8$ parallel attention layers, or heads.

$\large d_k = d_v = d_{model}/h=64$
