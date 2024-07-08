- [Run test](#run-test)
- [Gated Linear Attention Layer](#gated-linear-attention-layer)
- [Chunked Fuse 的实现](#chunked-fuse-的实现)
  - [1. fwd\_decay\_cumsum](#1-fwd_decay_cumsum)
  - [2. prepare\_qg\_kg](#2-prepare_qg_kg)
  - [3. fused\_chunk\_gla\_fwd\_kernel](#3-fused_chunk_gla_fwd_kernel)
  - [4. fwd\_inner\_chunk](#4-fwd_inner_chunk)

# Run test

```bash
python3 main.py
```

# Gated Linear Attention Layer

下面四行公式是gated linear attention layer在算法设计上得到的最终模型：

<p align="center">
<img src="figures/gla_equation.png" width=70%><br>
Fig 1. GLA layer以recurrent模式单步计算的线性代数公式<br>
</p>

下图是GLA layer的实现转换为tensor operator之间的数据流依赖。**图中GLA圈出来的部分实现了论文中的公式（8）和（9），但分成了多个kernel。**

<p align="center">
<img src="figures/gated_linear_attention_layer.png" width=60%><br>
Fig 2. GLA layer的数据流依赖
</p>

1. $\text{LogSigmoid}(x) = \log \frac{1}{1+\exp (-x)}$。将sigmoid函数的输出映射到log空间，将连乘转换为对数空间的连加。

    > *从下面的计算过程看，PyTorch这里的对数是以2位为底的对数。 $通过换底公式，为这个输出乘以 $\frac{1}{\ln 2}$ 转换为以 $e$ 为底的对数 （\log_2\left(x\right) = \frac{\ln x}{\ln 2}$ ）*。

2. $\text{silu}(x) = x * \text{sigmoid}(x)$

<p align="center">
<img src="figures/gla_data_accessed.png" width=60%><br>
Fig 3. GLA layer以recurrent模式单步计算单时间步访问的数据<br>
</p>

**从DNN模型设计的角度看GLA的数据流动，上图中蓝色虚线这一枝信息的流动起到了和传统RNN中input gate类似的功能，也是GLA这个模型中"gated"所指的部分。绿色虚线这一枝信息的流动起到了和传统RNN中output gate类似的功能。** GLA中的gating factor是一个和输入$K$大小相同的tensor，也就是对$K$的每个维度都进行不同强度的gating。

我们把GLA看做是一个RNN layer（causal形式，不attend到未来时刻，$i$时刻只去attend $i$时刻之前）由以下计算得到：

>for $i \in \left[0, L - 1 \right)$
>
> $\quad v_1 = q_i * \gamma * h_{t-1}*\exp(g_{k_i})$    // 对状态进行衰减
> 
> $\quad v_2 = k_i \otimes v_i$    // 当前时间步的输入
> 
> $\quad o_i = \text{sum}(v_1 + v_2, \text{dim}=-2)$    // 以上两步叠加，状态是2D的，进行reduce压缩到1D

# Chunked Fuse 的实现

分成了5步，前4步是4个triton kernel，最后一步是一个简单的相加，用了PyTorch的operator。下面的符号都尽量沿用了代码中对应的变量名，以便和代码对应。

|Notation|Explanation|取值量级||
|:--:|:--|:--|:--|
|$B$|batch size|32|
|$L$|sequence length|2048+|
|$H$|head number|4|
|$D_{qk}$|query和key的hidden dimension|1024或者2048这样的量级|
|$D_{v}$|value的hidden dimension|1024或者2048这样的量级|
|$BT$|序列长度维度上的分块大小|*固定取16*|太小了|
|$BK$|$D_{qk}$维度上的分块大小|$D_{qk}$和64中的较小值|
|$NK$|$NK=\frac{D_{qk}}{BK}$|$D_{qk}$维度上的分块数目|
|$BV$|$D_{v}$维度上的分块大小|$D_v$和64中的较小值|
|$NV$|$NV = \frac{D_v}{BV}$|$D_v$维度上的分块数目|

|输入tensor|形状|
|:--:|:--|
|$Q$|$[B, H, L, D_{qk}]$|
|$K$|$[B, H, L, D_{qk}]$|
|$V$|$[B, H, L, D_{v}]$|
|$gK$|$[B, H, L, D_{qk}]$|

下面表格第3列的”数据划分“，就对应了CUDA device kernel launch config中blocks的三个维度，也就是并发blocks数目。

|No.|Kernel|数据划分|theads per CTA|
|:--:|:--|:--|:--|
|1|$g_o=\text{fwd\_decay\_cumsum}(g)$|$NK,\frac{L}{BT}, B*H$|32|
|2|$q_g, k_g=\text{prepare\_qg\_kg}(q,k,g_o)$|$NK,\frac{L}{BT}, B*H$|32|
|3|$o = \text{fused\_chunk\_gla\_fwd\_kernel}(q_g,k_g,v,g_o,o)$|$NK, NV, B * H$|64|
|4|$o_2 = \text{fwd\_inner\_chunk}(q,k,g_o)$|$NK,\frac{L}{BT}, B * H$|128|
|5|$v_2 = \text{rearrange}(v, \text{'b h (n c) d'} \rightarrow \text{'b h n c d'}, n=\text{num\_chunk})$<br>$o = o+ o_2@v_2$|/|combine inner and intra chunks<br>由PyTorch operator完成|

## 1. fwd_decay_cumsum

第1个kernel是一个很小的element-wise kernel，用来计算沿着序列长度$L$维度的gated factor的累乘。输入的是通过低秩方法得到的gating factor $GK_{[B,H,L,D_{qk}]}$。由于多样本和多head之间完全独立，我们总是可以忽略$B$，$H$这两个全并行维度，他们最终会映射到CUDA的blocks之上并发进行处理。于是我们始终只关注如何处理一个序列。

Fig 4是fwd_decay_cumsum这个kernel处理数据的示意图，这个kernel在$G$的hidden维度和序列长度维度并行。

1. 每一个CTA相互独立地处理$D_{qk} \times BT$大小的数据。
2. 这个kernel的pattern是一个2D kernel，在序列长度维度上进行scan，在$G$的hidden dimension上没有数据依赖，可以全并发处理。

<p align="center">
<img src="figures/fwd_decay_cumsum.png" width=50%><br>
Fig 4. fwd_decay_cumsum的并行方式
</p>

这个kernel内部带有一个pattern为scan的for循环 ：$Y=\text{scan}\left((\vec{s}, \vec{x})\rightarrow f, I=\vec{1}, \text{rows}(X)\right)$，$f(\vec{s},\vec{x})$是scan携带的二元算符，公式如下：

$$f(\vec{s},\vec{x}) = \vec{s} + \vec{x} / \ln 2 $$

我们可以看到这个kernel**可能是为了提高并行性**，在蓝色小块内部进行了累积和运算，但是小块之间是独立的，也就是说这个kernel计算完毕后，**gating factor在每一个长度为16的local窗口内进行了累积，但是并没有在全序列长度范围内进行累积**。

## 2. prepare_qg_kg

以$Q$，$K$，$V$和$G$为输入，以和$Q$，$K$等大的$Q_g$和$K_g$为输出，完成为$Q$和$K$都乘以gating factor的作用。

这个kernel和`fwd_decay_cumsum`的分数据方式，kernel内部循环方式完全一样，也就是每个kernel依然独立地去处理Fig 4中蓝色部分的数据块。

来看这一步计算`last_decay`的指针偏移计算：
```python
last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k * BK + tl.arange(0, BK))
```

取到的数据如下图所示：

<p align="center">
<img src="figures/last_decay.png" width=30%>
</p>

当前kernel需要读取gating factor矩阵$G_k$中$D_{qk}$行，$BT$列大小的一块数据（Fig 4蓝色方块大小）的一块数据，而`last_decay`取到这块数据的最后一列。

这个kernel完成的数学计算如下：

$q = q * 2^{g} * \gamma$ $\qquad\leftarrow$ 这里2的幂次之后得到的是sigmoid的输出。
$k = k * 2^{\text{last\_decay} - g}$

也就是给$Q$和$K$都乘以gating factor $G$，进行衰减。

## 3. fused_chunk_gla_fwd_kernel

这个kernel的分数据方案如下，kernel 内部有一个for在序列长度维度进行循环。

<p align="center">
<img src="figures/fused_chunk_gla_fwd_kernel.png" width=40%><br>
Fig 5. fused_chunk_gla_fwd_kernel的并行方式
</p>

这个kernel完成的数学计算：

$$
\begin{align*}
o &= q \otimes h_{t-1}\\
h_t &= h_{t-1} * g_{db} + k\otimes v
\end{align*}
$$

上面第2个公式中的$g_{db}$对应代码中以下部分的数据：

```cpp
p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
```

$g_{db}$在初始时刻，取到了一个序列第一个分块（跨16列）的最后一列。随着kernel内的循环在序列长度分块上移动，$g_{db}$总是指向当前分块的最后一列。类似于`prepare_qg_kg`中`last_deacy`的作用。

## 4. fwd_inner_chunk

这个kernel分数据的方式和kernel 1，2 完全相同。每个小分块之间完全的独立。

<p align="center">
<img src="figures/fwd_inner_chunk.png" width=30%><br>
Fig 6. fwd_inner_chunk的并行方式
</p>

kernel内部的for循环在序列长度方向循环。