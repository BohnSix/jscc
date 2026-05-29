# Nonlinear Transform Source-Channel Coding for Semantic Communications

Pytorch Implementation of JSAC 2022 Paper "Nonlinear Transform Source-Channel Coding for Semantic Communications"

Arxiv Link: https://arxiv.org/abs/2112.10961

Project Page: https://semcomm.github.io/ntscc/

## Prerequisites
* Python 3.8 and [Conda](https://www.anaconda.com/)
* CUDA 11.0
* Environment
    ```
    conda create -n $YOUR_PY38_ENV_NAME python=3.8
    conda activate $YOUR_PY38_ENV_NAME
    
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install -r requirements.txt
    ```
  
## Usage


Example of test the PSNR model:
```bash
python main.py --phase test --checkpoint path_to_checkpoint
```

## Pretrained Models

Pretrained models (optimized for MSE) trained from scratch using randomly chose 500k images from the OpenImages dataset.

* Download [NTSCC w/o z models](https://drive.google.com/drive/folders/1qNRu_08-O5-lkqo3Sht48FCLITqqK6t-?usp=sharing) and put them into ./checkpoints folder.

Other pretrained models will be released successively.

Note: We reorganize code and the performances are slightly different from the paper's.

>  RD curves on [Kodak](http://r0k.us/graphics/kodak/), under AWGN channel SNR=10dB.
![kodak_rd](fig/kodak_results_bandwidth/kodak_psnr_cbr_10dB.png)

## Citation
If you find the code helpful in your research or work, please cite:
```
@ARTICLE{9791398,
  author={Dai, Jincheng and Wang, Sixian and Tan, Kailin and Si, Zhongwei and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Nonlinear Transform Source-Channel Coding for Semantic Communications}, 
  year={2022},
  volume={40},
  number={8},
  pages={2300-2316},
  doi={10.1109/JSAC.2022.3180802}
  }
```

## Acknowledgements
The NTSCC model is partially built upon the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [CompressAI](https://github.com/InterDigitalInc/CompressAI/). We thank the authors for sharing their code.




先给一句总括：

> **这篇论文的核心思想，是不再把图像直接映射为信道符号，而是先学习一个更适合传输的潜在表示，再通过熵模型估计每个局部信息量，按内容复杂度自适应分配传输资源，并利用超先验作为边信息辅助解码。**

这句话基本就是全篇主线。

---

## 🧭 一、论文背景：它为什么要做 NTSCC

这篇文章的出发点很明确：**标准 deep JSCC 在小图像上效果不错，但到高分辨率图像和高 CBR 场景时，性能会明显掉队。**

### 1. 标准 deep JSCC 的问题
传统 deep JSCC 大致做法是：

- 输入图像 $x$
- 用神经网络编码成连续信道输入 $s$
- 经 AWGN / 衰落信道
- 解码恢复图像 $\hat{x}$

这个框架有一个明显优点：  
它是端到端优化的，低 SNR 时相比传统分离式方案更鲁棒，也有 graceful degradation。

但作者指出，它有几个瓶颈：

- **高分辨率图像上性能下降快**
- **当 CBR 增大时，性能提升速度变慢**
- **不能像现代学习式压缩那样按内容复杂度灵活分配资源**
- **没有超先验 hyperprior 作为边信息辅助建模与解码**

### 2. 作者对问题的判断
论文的核心判断是：

> 标准 deep JSCC 的根本短板在于：**它不能很好识别源分布，因此难以做 patch-wise 的可变长度传输。**

这句话很关键。

更直白一点说：

- 简单区域（天空、墙面、背景）其实不需要很多传输维度
- 复杂区域（人物、水面、文字、纹理）需要更多资源
- 标准 deep JSCC 往往没有很好做到这点

结果就是：

- 简单 patch 很快“传满了”，多给的带宽没收益
- 复杂 patch 却可能资源不够
- 总体带宽利用不高

---

## 🔍 二、论文的核心目标

作者想做的是一个更强的端到端图像传输框架，满足三件事：

### 1. 先学到更适合传输的 latent 表示
不是直接传图像像素，而是先通过**非线性分析变换**得到潜在表示。

### 2. 对 latent 建立概率模型
通过**熵模型 + hyperprior**去估计每个局部表示的信息量。

### 3. 根据信息量做自适应传输
高熵 patch 多分资源，低熵 patch 少分资源，实现内容感知的变量长度传输。

于是就提出了：

## **NTSCC：Nonlinear Transform Source-Channel Coding**

它本质上是：

> **非线性变换编码（NTC） + deep JSCC + 超先验 + 自适应码率控制**

---

## 🏗️ 三、NTSCC 的整体框架

下面看整体结构。可以把它拆成两个大阶段：

### 阶段 A：非线性变换与概率建模
输入图像 $x$ 后，先经过：

- **分析变换 $g_a$**：得到 latent 表示 $y$
- **超先验分析变换 $h_a$**：得到超先验 $z$
- **超先验合成变换 $h_s$**：由 $\bar{z}$ 预测 $y$ 的分布参数，例如均值与方差

这一步类似现代 learned image compression 里的 hyperprior 压缩模型。

### 阶段 B：在 latent 空间做 JSCC
不是直接把 $x$ 送进 JSCC 编码器，而是把 $y$ 切成 patch 表示 $y_i$，然后：

- 根据信息熵为每个 $y_i$ 分配不同信道带宽成本
- 再用深度 JSCC 编码器 $f_e$ 把每个 $y_i$ 编码成连续信道符号 $s_i$
- 接收端用 $f_d^\star$ 恢复 latent，再用 $g_s$ 重建图像

所以整个流程是：

$
x \rightarrow g_a \rightarrow y \rightarrow \{h_a,h_s\} \text{建模分布} \rightarrow f_e \rightarrow \text{channel} \rightarrow f_d^\star \rightarrow g_s \rightarrow \hat{x}
$

这就是 NTSCC 的主干逻辑。

---

## 🧠 四、NTSCC 的核心创新点

这篇论文的贡献可以概括为三大块。

---

## 1️⃣ 非线性变换编码 + JSCC 的融合

### 传统 deep JSCC
传统 deep JSCC 是：

$
x \rightarrow f_e \rightarrow s \rightarrow \hat{s} \rightarrow f_d \rightarrow \hat{x}
$

### NTSCC
NTSCC 则是：

$
x \rightarrow g_a \rightarrow y \rightarrow f_e \rightarrow s \rightarrow \hat{s} \rightarrow f_d^\star \rightarrow \hat{y} \rightarrow g_s \rightarrow \hat{x}
$

区别就在于，先经过一个**非线性分析变换**把图像映射到潜在空间。

### 为什么这样更好
因为潜在空间中的表示：

- 更接近语义/统计上的有效表示
- 更容易建立熵模型
- 更容易估计每个 patch 的复杂度
- 更适合做自适应资源分配

这一步其实把“现代学习式压缩”的思想引入了 JSCC。

---

## 2️⃣ 自适应码率传输：内容复杂度决定传输资源

这是全篇最亮眼的创新之一。

### 核心思想
作者对每个 latent patch $y_i$ 建立条件熵模型：

$
P_{\bar{y}_i|\bar{z}}(\bar{y}_i|\bar{z})
$

通过这个条件概率模型，可以估计每个 patch 的熵，也就是“信息密度”。

如果某个 patch 熵高，说明：

- 内容复杂
- 难压缩
- 更值得分配多一点传输维度

于是给它较高的信道带宽成本。

相反，简单区域就少给一点资源。

### 公式思想
论文中给出了每个 patch 的带宽分配规则，大致形式是：

$
\bar{k}_{y_i} = Q'(-\eta \log P_{\bar{y}_i|\bar{z}}(\bar{y}_i|\bar{z}))
$

其中：

- $-\log P_{\bar{y}_i|\bar{z}}$ 表示 patch 的信息量/熵
- $\eta$ 是从熵映射到信道带宽成本的缩放因子
- $Q'$ 是离散化函数，把连续值映射到一组可选的码率等级

### 直观意义
NTSCC 不再“所有 patch 同等对待”，而是：
- 复杂 patch：多给资源
- 简单 patch：少给资源

这就实现了 **patch-wise variable-length transmission**。

论文中的可视化也验证了这一点：
- 水、文字、人等复杂区域分到更多带宽
- 天空、背景、平坦区域分到更少带宽

很像一个预算有限但头脑清醒的项目经理。

---

## 3️⃣ Hyperprior-aided codec refinement

这是另一个关键创新。

### 什么是 hyperprior
hyperprior 就是关于 latent 分布的更高层先验信息。  
NTSCC 中：

- $z = h_a(y)$
- $\bar{z}$ 量化后作为 side information
- $h_s(\bar{z})$ 预测 $y$ 的均值 $\bar{\mu}$ 和标准差 $\bar{\sigma}$

### 它的作用
这部分信息有两个用途：

#### 用途 A：发送端做码率分配
发送端利用 $\bar{z}$ 推断每个 patch 的熵，决定该分多少传输带宽。

#### 用途 B：接收端做解码 refinement
接收端不仅有 noisy channel output，还能利用 $\bar{\mu}, \bar{\sigma}$ 作为先验，辅助恢复 latent。

这可以减少：
- 训练集分布与某个具体样本之间的失配
- 信道噪声导致的恢复不确定性

### 为什么重要
普通 deep JSCC 解码器常常是“硬猜”。  
NTSCC 则相当于告诉解码器：

> 这个 patch 原本大概长什么样、方差多大、波动范围如何。

这明显更有利于恢复。

---

## 📐 五、论文的理论建模：变分率失真框架

这篇文章不是纯工程堆模块，理论上它把 NTSCC 放进了一个**VAE / 变分推断**的框架里分析。

### 1. 先回顾 NTC 的率失真建模
对于非线性变换编码 NTC，目标是最小化：

- latent 的编码代价
- side information 的代价
- 重建失真

对应一个标准的率失真目标。

### 2. NTSCC 的建模
在 NTSCC 中，除了压缩问题，还引入了真实信道传输。

最终可以把优化目标理解为三部分：

- **side information rate**：传 $\bar{z}$ 的代价
- **transmission rate**：传主信息 $s$ 的代价
- **distortion**：最终重建图像失真

本质上就是一个联合率失真优化问题：

$
\text{Loss} = \text{主信息传输代价} + \text{边信息代价} + \text{重建失真}
$

作者通过 KL 散度最小化把它写成变分形式，从而把整套系统放到了统一的理论框架下。

### 直观理解
这意味着 NTSCC 不是“拍脑袋加几个模块”，而是：

> **在一个统一的概率建模与率失真最优框架下，把压缩、先验建模和信道传输联合起来。**

这点很加分。

---

## 🏛️ 六、网络结构设计

这篇文章的网络设计有几个有意思的点。

---

## 1. 分析变换 $g_a$ 和合成变换 $g_s$

### 结构
作者用的是 **Transformer / Vision Transformer 风格结构**：

- 图像切分为 patch
- patch embedding
- 多层 Transformer block
- patch merging / patch division
- 多 stage 层次结构

### 特点
- 小图像用较少 stage
- 高分辨率图像用更多 stage
- CLIC2021 这种大图像上，为降低复杂度还用了 **Swin Transformer 的 shifted-window self-attention**

### 意义
这说明作者已经意识到：
- 高分辨率图像需要更强的特征建模能力
- 普通 CNN 可能不够
- Transformer 更利于学习长距离依赖和层次表示

---

## 2. 超先验网络 $h_a, h_s$

这部分是卷积网络结构，用来：

- 从 $y$ 提取超先验 $z$
- 从 $\bar{z}$ 预测 $\bar{\mu}, \bar{\sigma}$

它承担的是**概率分布建模**的职责，而不是主语义表示学习。

---

## 3. JSCC 编码器 $f_e$ 和解码器 $f_d^\star$

这部分很有意思，因为它们必须支持**不同 patch 使用不同码率**。

### 如果直接粗暴做会怎样
一个很直接但笨重的方案是：

- 每一种码率训练一套独立 encoder/decoder

但这样：
- 训练开销巨大
- 参数量爆炸
- 忽略 patch 间上下文关系

### 作者的方案
作者用了**动态网络 + shared Transformer blocks**：

- 共享若干 Transformer block 提取特征
- 对每个 patch 加一个 **rate token**
- 用 rate token 告诉网络：这个 patch 现在用的是哪一个带宽等级
- 再通过轻量 FC 层把输出映射到对应长度

### 接收端
接收端先把不同长度的 noisy channel vector reshape 到统一维度，再加上对应 rate token，通过共享 Transformer block 解码。

### 优点
- 同一网络支持多码率
- 参数共享，复杂度可控
- 可以保留 patch 之间的上下文依赖

这设计很聪明，避免了“每个码率单独开一家店”。

---

## 🧪 七、训练方式

论文明确给出了训练过程，关键点有两个：

---

## 1. 先预训练 NTC
作者没有一开始就把所有模块随机初始化端到端训练，而是先训练 NTC 模型：

- $g_a, g_s, h_a, h_s$
- 不考虑信道误差
- 先把图像压缩与超先验建模训稳定

### 为什么这么做
因为如果直接把：
- 非线性变换
- 超先验
- 熵模型
- 动态码率
- 信道噪声
- JSCC 编解码

全部一起训，优化会非常不稳定。

所以先预训练 NTC，相当于先把 latent 空间打磨好。

---

## 2. 再训练完整 NTSCC
在预训练基础上：
- 初始化 $g_a, g_s, h_a, h_s$
- 随机初始化 JSCC 编解码器 $f_e, f_d^\star$
- 再做联合优化

训练损失大致是：

$
L = d(x,\hat{x}_{NTSCC}) + d(x,\hat{x}_{NTC}) + \lambda(k_y + k_z)
$

其中：
- $d(x,\hat{x}_{NTSCC})$：端到端传输后的失真
- $d(x,\hat{x}_{NTC})$：辅助 NTC 失真项，用来稳定训练
- $k_y$：主信息传输开销
- $k_z$：边信息开销

### 这一设计很重要
它说明训练不只是单纯最小化最终图像误差，而是联合考虑：

- 传输质量
- 压缩质量
- 主信息带宽
- 边信息带宽

这非常符合率失真优化逻辑。

---

## 📊 八、实验设置与结果

论文在实验部分做得比较完整。

### 数据集
- **CIFAR10**：小图像
- **Kodak**：中等分辨率
- **CLIC2021**：高分辨率
- 训练大图时还用 Open Images 数据集

### 对比方法
- 标准 **Deep JSCC**
- **BPG + LDPC**
- **BPG + Capacity**
- **NTC + LDPC**
- **NTC + Capacity**

### 指标
- **PSNR**
- **MS-SSIM**
- **LPIPS**

---

## 1. PSNR 结果
结论很清楚：

- NTSCC 在多数场景下优于标准 deep JSCC
- 在 Kodak 和 CLIC2021 这样的高分辨率图像上提升更明显
- 相比 BPG + LDPC 也有明显竞争力
- 接近 NTC + Capacity 这个近似上界

作者还给了 BD-CBR 和 BD-PSNR 指标，显示：
- 在高分辨率图像上，NTSCC 可显著节省带宽
- 标准 deep JSCC 在高分辨率下甚至出现带宽成本增加

### 一个值得注意的细节
在 CIFAR10 小图上，NTSCC 甚至略逊于 deep JSCC。  
作者解释得很诚实：

- 因为 side information 的相对开销在小图上不划算
- 自适应与 refinement 的收益可能被边信息成本抵消

这说明 NTSCC 并不是“全场通杀”，而是**更适合高分辨率、复杂内容场景**。

---

## 2. MS-SSIM 结果
在感知质量指标上，NTSCC 提升更明显，尤其在高分辨率场景。

作者指出：
- BPG 系列是针对平方误差优化的
- 在 MS-SSIM 这类感知指标上，学习式方法更有优势

从表格结果看，NTSCC 在 CLIC2021 上的带宽节省非常显著。

---

## 3. LPIPS 与感知优化
论文进一步引入 LPIPS 作为更接近人类视觉感知的指标，并在训练中结合 cGAN、MSE 和 LPIPS 构造感知损失。

结果显示：

- **NTSCC (Perceptual)** 在 LPIPS 上显著优于其它方法
- 视觉效果更自然
- 更少块效应和失真伪影

这也很符合语义通信的目标：  
不是一味逐像素复刻，而是更贴近人类感知质量。

---

## 4. 消融实验
消融实验做了两件事：

### 消融 A：去掉 hyperprior 传输
如果不把 $\bar{z}$ 发给接收端，只用它指导发送端分配码率，那么性能会下降，但仍优于普通 deep JSCC。

说明：
- hyperprior 确实有帮助
- 但它不是绝对必须
- 在工程上可以根据带宽和复杂度灵活选择

### 消融 B：去掉 rate adaptation
如果不给 patch 分配不同码率，而统一使用同样带宽，系统退化成一个 ViT backbone 的 deep JSCC。

结果表明：
- 仅 backbone 升级就有收益
- 加上 rate adaptation 后收益进一步提升

这说明 NTSCC 的提升来自两个方面：
1. 更强的表示 backbone  
2. 更聪明的内容自适应资源分配

---

## ✅ 九、这篇论文的主要贡献，怎么一句句说清楚

如果你要做汇报，可以把贡献概括成下面四点：

### 贡献 1
提出了 **NTSCC** 框架，把非线性变换编码与 deep JSCC 融合起来。

### 贡献 2
引入 **条件熵模型** 对 latent patch 的复杂度建模，实现 patch-wise 自适应变量长度传输。

### 贡献 3
引入 **hyperprior-aided codec refinement**，把超先验作为边信息辅助接收端恢复 latent。

### 贡献 4
在 PSNR、MS-SSIM、LPIPS 等多个指标上，在不同分辨率图像上验证了优于 deep JSCC 与传统分离式方案的性能。

---

## ⚖️ 十、这篇论文的优点与局限

### 优点
这篇文章的优点很明显：

- **把 learned compression 和 deep JSCC 真正融合**
- **内容感知的资源分配设计很自然、也很有效**
- **理论和系统设计较完整**
- **感知指标与视觉结果都比较强**
- **高分辨率图像场景下优势突出**

### 局限
当然也有代价：

#### 1. 系统复杂度高
它不是一个简单模型，包含：
- 主 analysis/synthesis transform
- hyperprior 网络
- 熵模型
- 自适应码率模块
- JSCC 编解码
- side information 传输链路

训练与部署都比较复杂。

#### 2. side information 有成本
超先验 $\bar{z}$ 不是白送的。  
在小图上，它的开销会很显著。

#### 3. 码率上限受表示维度限制
论文也提到，输出 channel dimension $c=256$ 限制了 patch 最大可支持的传输维度，在超高 CBR 区域会出现饱和。

#### 4. 工程落地难度相对更高
相比一个直接端到端的 deep JSCC，NTSCC 更像一套“系统工程”，实现门槛更高。

---

## 💡 十一、怎么理解这篇论文的意义

这篇文章的真正意义，不只是“又做了一个更强的模型”，而是它指出了一条很清晰的路线：

> **未来的语义通信，不应只是把源数据端到端扔进信道，而应先学到有结构、可建模、可感知调度的潜在表示，再根据信息内容和统计复杂度进行资源分配。**

也就是说，它把语义通信从：

- “直接神经网络传输”

推进到了：

- “表示学习 + 概率建模 + 资源分配 + 联合传输”

这就是它的学术价值所在。

---

## 📝 十二、如果你要做汇报，可以这样总结

### 30 秒版本
> 这篇论文提出了 NTSCC 框架，将非线性变换编码与深度联合源信道编码结合。其核心思想是先将图像映射到潜在空间，再利用超先验和条件熵模型估计每个 patch 的信息复杂度，并根据信息量自适应分配传输带宽，同时在接收端利用超先验边信息进行解码 refinement。实验表明，该方法在高分辨率图像和多种感知指标上优于标准 deep JSCC 和传统分离式方案。

### 10 秒版本
> **NTSCC = 非线性潜表示学习 + 熵模型驱动的自适应 JSCC + hyperprior 辅助解码。**

