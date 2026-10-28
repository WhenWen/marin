# d512：原缩放 Muon 与 Moonlight 的最优配置

代码基线：

| 项目 | 值 |
|---|---|
| Repository | `marin-community/marin` |
| Branch | `fasttrack-moe-kma` |
| Commit | `4eba59cafa27a1ea174d1b5d6a28d9d1660eba10` |

Commit message：

```text
grug: address lint review (shared KDA fp32 dot, CPU-device test runner)

Share one fp32-accumulated dot between the KDA prep and state kernels, and run the multi-device CPU
checks through levanter.testing.cpu_devices.run_on_cpu_devices.
```

以下列出两组完整的 Adam 类参数设置，以及相对 d512 MuonH 基线的 Muon 更新变更。其余设置沿用 MuonH。
“最优”按各组已测试 LR 中的最低最终千问验证 loss 选择，均为 seed 0。

| 配置 | 最优 Muon 峰值基础 LR | 最终验证 loss |
|---|---:|---:|
| 原缩放 Muon，WD=0.1 | `0.015857567136604373` | **2.783604** |
| Moonlight，联合 LR | `0.007928783568302187` | **2.788950** |

**两组的 MoE router、`attn_gate` 和 AttnRes query 都使用普通 Adam，不使用 AdamH。**
router 和 `attn_gate` 有解耦 WD；AttnRes query 的 WD=0。
**原缩放 Muon 仅 LM head 使用 AdamH、WD=0；Moonlight 完全不使用 AdamH，LM head 使用 AdamW、WD=0.1。**

下表覆盖两组全部 Adam / AdamH / AdamW 参数。参数名省略层堆叠前缀；
同名参数在 MLA 层与 KDA 层中均存在时，两类层都适用该行规则。

| 参数 | 原缩放 Muon：优化器 | WD | Moonlight：优化器 | WD |
|---|---|---|---|---|
| LM head：`output_proj` | **AdamH** | **0** | **AdamW** | **0.1** |
| Token embedding：`token_embed` | Adam | 0 | Adam | 0 |
| Embedding / final RMSNorm：`embed_norm.weight`、`final_norm.weight` | Adam | 0 | Adam | 0 |
| 层内 RMSNorm：`rms_attn.weight`、`rms_mlp.weight` | Adam | 0 | Adam | 0 |
| MLA KV latent norm：`attn.kv_latent_norm.weight` | Adam | 0 | Adam | 0 |
| MoE latent norm：`mlp.latent_norm.weight` | Adam | 0 | Adam | 0 |
| KDA output norm：`attn.o_norm.weight` | Adam | 0 | Adam | 0 |
| MoE router 权重：`mlp.router` | **Adam + 解耦 WD** | `wd_t` | **Adam + 解耦 WD** | `wd_t` |
| Attention gate：`attn.attn_gate` | **Adam + 解耦 WD** | `wd_t` | **Adam + 解耦 WD** | `wd_t` |
| Router bias：`mlp.router_bias` | Adam | 0 | Adam | 0 |
| AttnRes attention query：`attn_res_query_attn` | **Adam** | **0** | **Adam** | **0** |
| AttnRes MLP query：`attn_res_query_mlp` | **Adam** | **0** | **Adam** | **0** |
| AttnRes final query：`attn_res_query_final` | **Adam** | **0** | **Adam** | **0** |
| Attention 分支卷积：`sconv_attn.weight` | Adam | 0 | Adam | 0 |
| MLP 分支卷积：`sconv_mlp.weight` | Adam | 0 | Adam | 0 |
| KDA Q/K/V 卷积：`attn.sconv_q.weight`、`attn.sconv_k.weight`、`attn.sconv_v.weight` | Adam | 0 | Adam | 0 |
| Inkling 相对位置参数：`attn.rel_pos.r_proj`、`attn.rel_pos.proj` | Adam | 0 | Adam | 0 |
| KDA forget-gate 投影：`attn.w_a_down`、`attn.w_a_up` | Adam | 0 | Adam | 0 |
| KDA 衰减参数：`attn.a_log` | Adam | 0 | Adam | 0 |
| KDA 时间步偏置：`attn.dt_bias` | Adam | 0 | Adam | 0 |

`wd_t = 0.02 * max(1 - t/2475, 0)`，其中 `t` 为从 0 开始的优化器更新计数。
这是 router / `attn_gate` 的解耦 WD 系数，从 0.02 线性退火至 0；实际收缩项还乘以该组当前 LR。

上表所有参数的峰值 LR 如下；AttnRes query 单独使用 0.1 倍 Adam LR。

| Adam 类参数范围 | 原缩放 Muon：峰值 LR | Moonlight：峰值 LR |
|---|---:|---:|
| LM head | `0.02242598651082779` | `0.007928783568302187` |
| 全部 AttnRes query：attention / MLP / final | `0.0005175227656344875` | `0.0007928783568302187` |
| 上表其余全部参数，包括 router 和 `attn_gate` | `0.005175227656344875` | `0.007928783568302187` |

表中的 Adam 均为普通 Adam，不含 Hyperball 范数约束；AdamH 使用 Adam 方向加范数缩放与球面投影。
AdamW 表示普通 Adam 加解耦 weight decay，router / `attn_gate` 也采用这一类解耦衰减规则。
GatedNorm 的 `w_down/w_up` 是 Muon 参数，不属于 RMSNorm scale 或上表的 Adam 参数，WD=0.1。

原缩放 Muon 将原 MuonH 参数组替换为普通 Muon，去掉按参数范数缩放和球面投影，
并对这些参数施加固定的解耦 WD=0.1。
Newton–Schulz 方向的形状缩放为 `sqrt(max(1, fan_out / fan_in))`。

| 变更的参数组 | 优化器 | 峰值 LR | WD |
|---|---|---:|---:|
| 注意力主矩阵、MoE routed/shared expert 矩阵、latent 投影 | Muon | `0.015857567136604373` | `0.1` |
| GatedNorm 的 `w_down/w_up` 矩阵 | Muon | `0.015857567136604373` | `0.1` |
| KDA `w_beta` | Muon | `0.031715134273208746` | `0.1` |

该配置只扫描 Muon LR；LM head 继续使用基线的 AdamH，普通 Adam 和 AdamH 的 LR 均保持基线值。

Moonlight 同样将原 MuonH 参数组替换为普通 Muon 并施加解耦 WD=0.1，
将 Newton–Schulz 方向的形状缩放改为 `0.2 * sqrt(max(fan_in, fan_out))`。
Muon、普通 Adam 与 head 共用基础 LR；head 从 AdamH 改为 AdamW，WD=0.1。

| 变更的参数组 | 优化器 | 峰值 LR | WD |
|---|---|---:|---|
| 注意力主矩阵、MoE routed/shared expert 矩阵、latent 投影 | Moonlight 缩放的 Muon | `0.007928783568302187` | 固定 `0.1` |
| GatedNorm 的 `w_down/w_up` 矩阵 | 同上 | `0.007928783568302187` | 固定 `0.1` |
| KDA `w_beta` | 同上 | `0.015857567136604373` | 固定 `0.1` |

两种 Muon 的解耦 WD 均按下面的规则执行。令 `U_t` 为经过动量、Newton–Schulz 和对应形状缩放后的方向：

```text
W_next = (1 - group_lr_t * 0.1) * W - group_lr_t * U_t
```

WD 系数固定为 0.1，实际收缩量随各组 LR 变化；形状缩放只作用于更新方向，不额外乘到 WD 上。

两组之间的 Muon 缩放、LR 配置及 head 更新规则均有差异，验证 loss 差异不能单独归因于其中一项。
