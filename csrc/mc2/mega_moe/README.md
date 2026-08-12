# MegaMoE

## Product support status

| Product | Support |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 training series products/Atlas A3 inference series products </term> | √ |
| <term>Atlas A2 training series products/Atlas A2 inference series products </term> | √ |
| <term>Atlas 200I/500 A2 inference product </term> | × |
| <term>Atlas inference series products</term> | × |
| <term>Atlas training series products </term> | × |

## Function description

- Operator function: The MegaMoE operator integrates the complete calculation process of the expert FFN of the MoE layer and the front and rear data communication (i.e. Dispatch + Linear1 + SwiGLU + Linear2 + Combine) into a single operator, realizing the masking of communication and calculation.

- Calculation formula:
  - Input:
    - $\mathbf{X} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$: activation matrix, corresponding to the input parameter `x`. $\text{totalNumTokens}$ is the global total number of tokens, and $\text{hidden}$ is the hidden layer dimension.
    - $\mathbf{E} \in \mathbb{Z}^{\text{totalNumTokens} \times \text{topK}}$: Expert number matrix selected by token, corresponding to the input parameter `topkIds`. $\text{topK}$ is the number of experts selected for each token.
    - $\mathbf{G} \in \mathbb{R}^{\text{totalNumTokens} \times \text{topK}}$: The gating weight matrix of the expert selected by token, corresponding to the input parameter `topkWeights`.
    - $\mathbf{W}_1^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{hidden} \times (2 \text{intermediateHidden})}$: Linear1 weight of the routing MoE expert, corresponding to the MoE expert part of the input parameter `weight1`.
    - $\mathbf{W}_2^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{intermediateHidden} \times \text{hidden}}$: Linear2 weight of the routing MoE expert, corresponding to the MoE expert part of the input parameter `weight2`.
    - $\mathbf{W}_1^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{hidden} \times (2 \text{intermediateHidden})}$: Linear1 weight of shared experts, corresponding to the input parameter `sharedWeight1`.
    - $\mathbf{W}_2^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{intermediateHidden} \times \text{hidden}}$: Linear2 weight of shared experts, corresponding to the input parameter `sharedWeight2`.
  - Output:

    - $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$: The final output matrix corresponds to the output parameter `y`.
  - Conventions:
    - $⋅$ represents matrix multiplication, and $⊙$ represents element-wise multiplication.
    - $\left \lfloor z\right \rceil$ means round $z$ to the nearest whole number, and $\left \lfloor z\right \rfloor$ means round $z$ down.
    - $|z|$ means taking the absolute value, $\max(z)$ means taking the maximum value.
    - The set of all tokens is $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{totalNumTokens} - 1\} \}$.
    - The token representation (that is, the hidden-state vector) of $\text{token}_i$ is $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$, and $\mathbf{x}_i = \mathbf{X}[i,:]$.
    - The expert index for $\text{token}_i$ is $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{topK} - 1\},\quad e_{i,k} \in \{0,\dots,\text{moeExpertNum} - 1\}$.
    - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$. The superscript $\text{sym}$ of $\mathbb{Z}_8^{\text{sym}}$ represents the symmetric quantization value range interval: its value range is symmetrically rounded with respect to $-127$ and $127$, which is different from the $[-128, 127]$ value range of standard INT8, so it is $\text{sym}$ superscript distinction.
    - The tensor slicing operation uses the Python-style `start:stop:step` notation, for example, $[0::2, :]$ represents taking even-numbered rows, and $[1::2, :]$ represents taking odd-numbered rows.
    - $\mathrm{bitcast}_{T}(\mathbf{Z})$ represents a binary reinterpretation operation, which reinterprets the underlying binary data of tensor $\mathbf{Z}$ according to the target type $T$.
  - <span id="activation-formulas">Activation formulas:</span>
    - Note that the gate branch after splitting the Linear1 output is $\mathbf{G}$, the up branch is $\mathbf{U}$, and the activation output is $\mathbf{A}$. The Sigmoid function and SiLU function are defined as:
      $$
      \sigma(z)=\frac{1}{1+e^{-z}}, \qquad
      \operatorname{SiLU}(z)=z\cdot\sigma(z)=\frac{z}{1+e^{-z}}.
      $$
      Let the clipping value be $c$. When clipping is disabled (not configured or configured as $0$), mathematically $c=+\infty$. Define
      $\mathbf{G}_c=\min(\mathbf{G},c)$ and $\mathbf{U}_c=\operatorname{clip}(\mathbf{U},-c,c)$.
    - `swiglu`:
      $$
      \mathbf{A}=\operatorname{SwiGLU}(\mathbf{G},\mathbf{U})
      =\operatorname{Swish}_1(\mathbf{G}_c)\odot\mathbf{U}_c
      =\operatorname{SiLU}(\mathbf{G}_c)\odot\mathbf{U}_c.
      $$
    - `swiglustep`:
      $$
      \mathbf{A}=\operatorname{SwiGLUStep}(\mathbf{G},\mathbf{U})=\min\!\left(\operatorname{SiLU}(\mathbf{G}),c\right)\odot\mathbf{U}_c.
      $$
    - `swigluoai`:
      $$
      \mathbf{A}=\operatorname{SwiGLUOAI}(\mathbf{G},\mathbf{U})=\mathbf{G}_c\odot\sigma(\alpha\mathbf{G}_c)\odot(\mathbf{U}_c+\beta).
      $$
    - `situglu`:
      $$
      \mathbf{A}=\operatorname{SiTUGLU}(\mathbf{G},\mathbf{U})=\beta\tanh\!\left(\frac{\mathbf{G}}{\beta}\right)\odot\sigma(\mathbf{G})\odot L(\mathbf{U};\beta_{\mathrm{linear}}),
      $$
      where
      $$
      L(\mathbf{U};\beta_{\mathrm{linear}})=
      \begin{cases}
      \mathbf{U}, & \text{linear\_beta not configured},\\
      \beta_{\mathrm{linear}}\tanh\!\left(\dfrac{\mathbf{U}}{\beta_{\mathrm{linear}}}\right),
      & \text{linear\_beta configured}.
      \end{cases}
      $$
- Calculation process

    The data types of activation matrix A and weight matrix W supported by each product during Linear calculation are as follows:

    | Scene name | A | W |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT8_E4M3FN, FLOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    - <term>Atlas A2 training series products/Atlas A2 inference series products </term>, <term>Atlas A3 training series products/Atlas A3 Inference series product </term>: does not support the A8W8-FP scenario in the above table.
    - <term>Ascend 950PR/Ascend 950DT</term>: The A16W16, A8W8-INT, and A8W4-INT scenarios in the above table are not supported.

    <details>
    <summary> A16W16 non-quantized scene </summary>

    - **EP Dispatch**

        In the Dispatch phase, each $\text{token}_i$ sends its token representing $\mathbf{x}_i$ to the expert $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$. That is, for each $k$, expert $e_{i,k}$ receives one copy of $\mathbf{x}_i$.

        records $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ as the set of token indexes assigned to expert $e$. The size of the set $I_e$ is $N_e = |I_e|$, which is the expert $e$. The total number of tokens that need to be processed is $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$, which is a matrix composed of all token representations $\mathbf{x}_i$ ** stacked in any fixed order**. This matrix is the representation of all tokens received by expert $e$ after Dispatch.

        For each $\text{token}_i$ and its selected expert $e_{i,k}$, there is a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$. This mapping records the position of $\mathbf{x}_i$ in the input matrix $\mathbf{X}_{e_{i,k}}$ of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feed-forward network (FFN), using the **SwiGLU** structure to improve expression capabilities. The entire calculation process is divided into the following three sub-steps.

        **1. Linear1 projection**

        Linear1 projection is the first layer of linear transformation of the expert network, which simultaneously produces the pre-activation values required by the **gate part** and the **up part**. Its calculation formula is
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}\bigr)
        $$
        **2. Activate**

        Split $\mathbf{H}_e$ into the gate branch $\mathbf{G}_e$ and up branch $\mathbf{U}_e$ along the column dimension, and calculate the intermediate activation representation $\mathbf{A}_e$ according to `activation`. For each supported activation type, see [Activation formulas](#activation-formulas).

        **3. Linear2 projection**

        Linear2 projection acts as a second-level linear transformation, projecting the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, making the expert output compatible with subsequent operations such as residual concatenation.
        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        After the above calculation, each line of output of expert $e$ corresponds to a token in its batch. For $\text{token}_i$, its output line in the expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.
    - **Token Combine**

        Combine is responsible for collecting the output vectors calculated by all experts, performing a weighted sum according to the expert weight originally assigned to each token, and finally generating a fused output for each token. Using the previously recorded position index $\operatorname{row}(i,k)$, the rows belonging to $\text{token}_i$ are retrieved from the output matrix of expert $e_{i,k}$, multiplied by the gating weights and summed:

        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ to expert $e_{i,k}$.

        The output of all tokens is stacked in the order of input to the final output $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$.
    </details>

    <details>
    <summary> A8W8-INT Quantization scene </summary>

    - **Input**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$: Channel-by-channel scaling factor of Linear1 weight matrix, corresponding to the input parameter `weightScales1`.
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$: Channel-by-channel scaling factor of Linear2 weight matrix, corresponding to the input parameter `weightScales2`.

    - **EP Dispatch**

        Before Dispatch communication, the original BF16 activation matrix $\mathbf{X}$ is first quantized to INT8. For each $\text{token}_i$, calculate its per-token scaling factor:
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        is then quantized to get INT8 representation:
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        During the Dispatch communication phase, each $\text{token}_i$ sends its quantized vector $\mathbf{q}_i$ and scaling factor $s^{X}_i$ to the expert $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$ ($e_{i,k} = \mathbf{E}[i,k]$).

        records $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ as the set of token indexes assigned to expert $e$. The size of the set $I_e$ is $N_e = |I_e|$, which is the expert $e$. The total number of tokens that need to be processed. Then $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ is a matrix composed of all $\mathbf{q}_i$ that satisfy $i \in I_e$ **stacked in any fixed order**. This matrix is the representation of all the tokens received by expert $e$ after Dispatch; similarly, the corresponding expert The scaling factor vector received by $e$ is denoted as $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$, whose elements are stacked by all $s^{X}_i$ satisfying $i \in I_e$ in the same row order as $\mathbf{Q}_e$.

        For each $\text{token}_i$ and its selected expert $e_{i,k}$, there is a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$. This mapping records the position of $\mathbf{q}_i$ in the input matrix of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feed-forward network (FFN), using the SwiGLU structure to improve expression capabilities. In the A8W8 scenario, both linear layers use INT8 input and INT8 weights for matrix multiplication to obtain INT32 intermediate results and inverse quantization. Specifically divided into three sub-steps.

        **1. Linear1 projection (INT8 matrix multiplication + inverse quantization)**

        Linear1 projection is the first layer of linear transformation of the expert network, which simultaneously produces the pre-activation values required by the **gate part** and the **up part**. During calculation, INT8 matrix multiplication is performed and the INT32 calculation result is obtained:
        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$
        is then dequantized to the preactivation value $\mathbf{H}_e$:
        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **2. Activate**

        Split $\mathbf{H}_e$ into the gate branch $\mathbf{G}_e$ and up branch $\mathbf{U}_e$ along the column dimension, and calculate the intermediate activation representation $\mathbf{A}_e$ according to `activation`. For each supported activation type, see [Activation formulas](#activation-formulas).

        **3. Linear2 projection (quantization + INT8 matrix multiplication + inverse quantization)**

        Linear2 projection acts as a second-level linear transformation, projecting the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, making the expert output compatible with subsequent operations such as residual concatenation.
        In the A8W8 scenario, the activation value needs to be quantized to INT8, so the scaling factor is first calculated for each line (each token) of $\mathbf{A}_e$:
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        Gets the active scaling factor $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ for expert $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ during Linear2 calculations. Then quantify:
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$
        then performs INT8 matrix multiplication and inverse quantization:
        $$
        \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
        $$
        $$
        \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        After the above calculation, each line of output of expert $e$ corresponds to a token in its batch. For $\text{token}_i$, its output line in the expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine is responsible for collecting the output vectors calculated by all experts, performing a weighted sum according to the expert weight originally assigned to each token, and finally generating a fused output for each token. Using the previously recorded position index $\operatorname{row}(i,k)$, the rows belonging to $\text{token}_i$ are retrieved from the output matrix of expert $e_{i,k}$, multiplied by the gating weights and summed:
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ to expert $e_{i,k}$.

        The output of all tokens is stacked in the order of input to the final output $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$.

    </details>

    <details>
    <summary> A8W4-INT quantization scene </summary>

    - **Input**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$: Channel-by-channel scaling factor of Linear1 weight matrix, corresponding to the input parameter `weightScales1`.
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$: Channel-by-channel scaling factor of Linear2 weight matrix, corresponding to the input parameter `weightScales2`.
        - $\mathbf{B}_1 \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$: Linear1 offset matrix, generated offline by the INT4 quantization process, corresponding to the input parameter `bias1`.
        - $\mathbf{B}_2 \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$: Linear2 bias matrix, generated offline by the INT4 quantization process, corresponding to the input parameter `bias2`.

    - **EP Dispatch**

        Before Dispatch communication, the original BF16 activation matrix $\mathbf{X}$ is first quantized to INT8. For each $\text{token}_i$, calculate its per-token scaling factor:
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        is then quantized to obtain INT8 representation:
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        In the Dispatch communication phase, each $\text{token}_i$ sends its quantized vector $\mathbf{q}_i$ and scaling factor $s^{X}_i$ to the expert $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$

        records $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ as the set of token indexes assigned to expert $e$. The size of the set $I_e$ is $N_e = |I_e|$, which is the expert $e$. The total number of tokens that need to be processed. Then $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ is a matrix composed of all $\mathbf{q}_i$ **stacked in any fixed order** that satisfy $i \in I_e$. This matrix is the representation of all tokens received by expert $e$ after Dispatch; similarly, the corresponding expert The scaling factor vector received by $e$ is denoted as $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$, whose elements are stacked by all $s^{X}_i$ satisfying $i \in I_e$ in the same row order as $\mathbf{Q}_e$.

        For each $\text{token}_i$ and its selected expert $e_{i,k}$, there is a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$. This mapping records the position of $\mathbf{q}_i$ in the input matrix of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feed-forward network (FFN), using the **SwiGLU** structure to improve expression capabilities. In the A8W4-INT scenario, both linear layers use the MSD (Mixed-precision Split-activation Decomposition) scheme for matrix multiplication. By splitting the INT8 value into two signed INT4s of high 4 bits and low 4 bits, the INT8×INT4 matrix multiplication can be decomposed into two INT4×INT4 matrix multiplications, thereby utilizing the hardware INT4 matrix multiplication acceleration. The mathematical principles and implementation logic of this solution can be found in [GroupedMatmul W4A8 Quantization and the MSD Solution](https://gitcode.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81).

        **1. Generate the bias matrix for accuracy compensation (generated offline, completed outside the operator, and input as the operator)**

        MSD scheme is the binary reinterpretation splitting of the quantized INT8 activation value into two INT4 components. Since the low-order INT4 component is defined by $(\mathbf{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$, its value range is mapped to $[-8, 7]$, which is equivalent to subtracting 8 from the original unsigned low 4 bits (0~15). When the high and low bits of INT4 are matrix multiplied and merged with the weight matrix respectively, this offset will introduce a constant term that needs to be compensated in the final result.

        The original INT8 activation matrix is ​​$\mathbf{X}^{\text{int8}}$, the split high-bit and low-bit INT4 activation matrices are $\mathbf{X}_1^{\text{int4}}$ and $\mathbf{X}_2^{\text{int4}}$ respectively, and the weight matrix is ​​$\mathbf{W}$. The restored relationship is:

        $$
        \mathbf{X}^{\text{int8}} = 16 \times \mathbf{X}_1^{\text{int4}} + (\mathbf{X}_2^{\text{int4}} + 8 \cdot \mathbf{1}_{\text{mat}})
        $$

        where $\mathbf{1}_{\text{mat}}$ is an all-1 matrix of the same shape as $\mathbf{X}_2^{\text{int4}}$, used for element-wise addition of 8. Let $\mathbf{1}$ be an all-1 column vector with the same shape as the input feature dimension, then the matrix multiplication expansion is:

        $$
        \mathbf{X}^{\text{int8}} \cdot \mathbf{W} = 16 \cdot (\mathbf{X}_1^{\text{int4}} \cdot \mathbf{W}) + (\mathbf{X}_2^{\text{int4}} \cdot \mathbf{W}) + 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W})
        $$

        Here $\mathbf{1}^\top \cdot \mathbf{W}$ means summing each column of the weight matrix $\mathbf{W}$, and the result is a row vector (the shape is the same as the output dimension). In the specific implementation, the row vector will be broadcast to the batch dimension and added to the output of all tokens. It can be seen that if only the first two items are calculated, a positive term will be missing. In order to accurately restore the results, the compensation offset needs to be pre-calculated:

        $$
        \text{bias} = 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W}),
        $$

        $\mathbf{B}_1$ and $\mathbf{B}_2$ follow this method and use the corresponding all-1 column vectors (the dimensions are $\text{hidden}$ and $\text{hidden}$ respectively). $\text{intermediateHidden}$) are multiplied with their respective weight matrices to obtain shape-matched bias row vectors. These offsets are passed to the operator after the offline phase calculation is completed, and are directly involved in the addition when merging the high and low results, thereby eliminating the error introduced by the offset.

        **2. Linear1 projection (INT4 × INT4 matrix multiplication + inverse quantization)**

        Linear1 projection is the first layer of linear transformation of the expert network, which simultaneously produces the pre-activation values required by the **gate part** and the **up part**.

        **(1) Activate reinterpretation as INT4**

        Binary reinterpretation of the INT8 activation tensor $\mathbf{Q}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ into two INT4 views spliced alternately:

        $$
        \mathbf{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{hidden}}} \left( \mathbf{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{hidden}}
        $$

        where the high and low components are given directly by the even and odd lines of $\mathbf{Q}_e^{\text{int4}}$:

        $$
        \mathbf{Q}_e^{\text{high}} = \mathbf{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}, \quad
        \mathbf{Q}_e^{\text{low}} = \mathbf{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}
        $$

        Their values can be calculated from the original INT8 as follows:

        $$
        \mathbf{Q}_e^{\text{high}} = \left\lfloor \frac{\mathbf{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{Q}_e^{\text{low}} = (\mathbf{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        restores the relationship to $\mathbf{Q}_e^{\text{int8}} = 16\mathbf{Q}_e^{\text{high}} + (\mathbf{Q}_e^{\text{low}} + 8)$. Since $\mathrm{bitcast}$ only changes the type view, $\mathbf{Q}_e^{\text{int4}}$ shares the underlying physical memory with $\mathbf{Q}_e^{\text{int8}}$ without any data rearrangement or copying.

        **(2) INT4 × INT4 matrix multiplication and weight inverse quantization**

        Performs matrix multiplication of the reinterpreted INT4 activation view $\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ with the weight matrix $\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediateHidden}}$, and applies the weight scaling factor $\mathbf{s}^{W1}_e$ for inverse quantization to obtain the result:

        $$
        \mathbf{C}_e = \bigl( \mathbf{Q}_e^{\text{int4}} \cdot \mathbf{W}_1[e] \bigr) \odot \mathbf{s}^{W1}_e
        \;\in\; \mathbb{R}^{2N_e \times 2\cdot\text{intermediateHidden}}
        $$

        records the results in even rows and odd rows as the following matrix view, which correspond to the calculation results of $\mathbf{Q}_e^{\text{high}}$ and $\mathbf{Q}_e^{\text{low}}$:

        $$
        \mathbf{C}_e^{\text{high}} = \mathbf{C}_e[0::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}, \qquad
        \mathbf{C}_e^{\text{low}}  = \mathbf{C}_e[1::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **(3) Precision compensation and activation inverse quantization**

        uses the offset $\mathbf{B}_1[e]$ and the activation scaling factor $\mathbf{s}^{X}_e$ (row vector) to perform precision compensation and activation inverse quantization respectively. The final pre-activation value is:

        $$
        \mathbf{H}_e = \Bigl( 16 \cdot \mathbf{C}_e^{\text{high}} + \mathbf{C}_e^{\text{low}} + \mathbf{B}_1[e] \Bigr) \odot \mathbf{s}^{X}_e \quad\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        where $\odot \mathbf{s}^{X}_e$ means multiplying each row of the matrix by the corresponding scalar in $\mathbf{s}^{X}_e$.

        **3. Activate**

        Split $\mathbf{H}_e$ into the gate branch $\mathbf{G}_e$ and up branch $\mathbf{U}_e$ along the column dimension, and calculate the intermediate activation representation $\mathbf{A}_e$ according to `activation`. For each supported activation type, see [Activation formulas](#activation-formulas).

        **4. Linear2 projection (quantization + INT4 × INT4 matrix multiplication + inverse quantization)**

        Linear2 projection acts as a second-level linear transformation, projecting the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, making the expert output compatible with subsequent operations such as residual concatenation.
        In the A8W4 scenario, the activation value is first quantized into INT8, and then binary reinterpreted into a spliced ​​view of two INT4s, and the calculation is completed with one matrix multiplication. The process is as follows.

        **(1) Activate quantization**

        Calculate the scaling factor for each line of $\mathbf{A}_e$ and quantize to INT8:
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        gets the activation scaling factor $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ of expert $e$ during Linear2 calculation, and calculates the quantized result:
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$

        **(2) Activate reinterpretation as INT4**

        Binary reinterpretation of the INT8 activation tensor $\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}$ into two INT4 views spliced alternately:

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}
        $$

        where the high and low components are given directly by the even and odd lines of $\mathbf{A}_e^{\text{int4}}$:

        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}
        $$

        Their values can be calculated from the original INT8 as follows:

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        restores the relationship to $\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$. Since $\mathrm{bitcast}$ only changes the type view, $\mathbf{A}_e^{\text{int4}}$ shares the underlying physical memory with $\mathbf{A}_e^{\text{int8}}$ without any data rearrangement or copying.

        **(3) INT4 × INT4 matrix multiplication and weight inverse quantization**

        Performs matrix multiplication of the reinterpreted INT4 activation view $\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}$ with the weight matrix $\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediateHidden} \times \text{hidden}}$, and applies the weight scaling factor $\mathbf{s}^{W2}_e$ for inverse quantization to obtain the result:

        $$
        \mathbf{D}_e = \bigl( \mathbf{A}_e^{\text{int4}} \cdot \mathbf{W}_2[e] \bigr) \odot \mathbf{s}^{W2}_e
        \;\in\; \mathbb{R}^{2N_e \times \text{hidden}}
        $$

        records the results in even rows and odd rows as the following matrix view, which correspond to the calculation results of $\mathbf{A}_e^{\text{high}}$ and $\mathbf{A}_e^{\text{low}}$:

        $$
        \mathbf{D}_e^{\text{high}} = \mathbf{D}_e[0::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}, \qquad
        \mathbf{D}_e^{\text{low}}  = \mathbf{D}_e[1::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        **(4) Precision compensation and activation inverse quantization**

        uses the offset $\mathbf{B}_2[e]$ and the activation scaling factor $\mathbf{s}^{A_e}_e$ (row vector) to perform precision compensation and activation inverse quantization respectively. The final output is:

        $$
        \mathbf{Y}_e = \Bigl( 16 \cdot \mathbf{D}_e^{\text{high}} + \mathbf{D}_e^{\text{low}} + \mathbf{B}_2[e] \Bigr) \odot \mathbf{s}^{A_e}_e
        \quad\in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        where $\odot \mathbf{s}^{A_e}_e$ means multiplying each row of the matrix by the corresponding scalar in $\mathbf{s}^{A_e}_e$.

        After the above calculation, each line of output of expert $e$ corresponds to a token in its batch. For $\text{token}_i$, its output line in the expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine is responsible for collecting the output vectors calculated by all experts, performing a weighted sum according to the expert weight originally assigned to each token, and finally generating a fused output for each token. Using the previously recorded position index $\operatorname{row}(i,k)$, the rows belonging to $\text{token}_i$ are retrieved from the output matrix of expert $e_{i,k}$, multiplied by the gating weights and summed:
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ to expert $e_{i,k}$.

        The output of all tokens is stacked in the order of input to the final output $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$.

    </details>

    <details>
    <summary> A8W8-FP quantization scene </summary>

    In the first stage, the input tokens are collected according to expert groups and then quantified by MXFP8 to generate the quantified input and scaling factors of each expert:

    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
    $$

    Description: Tokens are collected sorted by experts according to `topkIds`. $\mathcal{T}_e$ is the Token assigned to expert $e$. Index set, $E_{local}$ represents the maximum number of tokens received by the current expert, the value of each expert may be different, $X[\mathcal{T}_e]$ is the corresponding sub-matrix. $\mathrm{Q}_{\text{MX}}$ represents MX group-by-group quantization (group size = 32), extracting the shared index for each group of 32 elements and quantizing them to the FP8 target type (FLOAT8_E5M2 or FLOAT8_E4M3FN), and outputting the FLOAT8_E8M0 scaling factor. The quantified data will serve as input to GMM1.
The second stage of

    performs GMM1 matrix multiplication (dividing $W_1$ into two halves along the column direction to calculate separately), SwiGLU activation and MX quantization for each expert:

    $$
    G_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(G)}, S_{1,e}^{(G)}), \quad U_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(U)}, S_{1,e}^{(U)})
    $$

    $$
    A_e = \operatorname{SwiGLU}(G_e,U_e)
    $$

    $$
    \hat{A}_e,\ S_{A,e} = \mathrm{Q}_{\text{MX}}(A_e)
    $$

    Description: Connect the front $N/2$ column $W_{1,e}^{(G)}$ and the rear $N/2$ column $W_{1,e}^{(U)}$ with MX respectively. The dequantized input is subjected to matrix multiplication to obtain the gate branch $G_e$ and the up branch $U_e$. For the calculation method of SwiGLU, see [Activation formulas](#activation-formulas); its output dimension is $N/2$. Then MX quantization is performed on the activation output to obtain the quantized input $\hat{A}_e$ of GMM2.

    The third stage performs GMM2 matrix multiplication for each expert and distributes the results according to the target Rank:

    $$
    O_e = \mathrm{DQ}_{\text{MX}}(\hat{A}_e, S_{A,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
    $$

    Description: Perform MX inverse quantization matrix multiplication between the quantized SwiGLU output and the second set of weights $W_2$, map the $N/2$ dimensional intermediate representation back to the $H$ dimensional hidden space, and obtain the output of each expert $O_e$. After the calculation is completed, the results are written to the remote end through RDMA peermem according to the expert offset address of the target Rank to achieve cross-Rank aggregation.

    When the shared expert is enabled (`sharedExpertNumPerRank` > 0), the shared expert locally performs the same GMM1 + SwiGLU + GMM2 calculation as the routing expert on each card for all tokens of the card, using `sharedWeight1`, `sharedWeight2`, `sharedWeightScales1`, `sharedWeightScales2` do not need to participate in Dispatch communication. The output of each shared expert is recorded as $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{sharedExpertNumPerRank} - 1\}$.

    In the fourth stage, all Tokens are weighted and summed according to the routing weight, and the shared expert output is superimposed to restore the output to the same shape as the input:

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)] + \sum_{s=0}^{\text{sharedExpertNumPerRank}-1} O^{\mathrm{shared}}_s[i]
    $$

    Description: For each Token $i$, take out the corresponding row from the aggregated expert results according to the sorted routing index $\pi(i,k)$, and add them element by element according to the weight in `topkWeights`. Then directly add the output of each shared expert on the current token to get the final output. $Y$. When shared experts are not enabled, the shared expert summation term is zero.

    Among them, $X$ represents the parameter `x`, $W$ represents the parameter `topkWeights`, and $W_1$ represents the parameter `weight1`, $W_2$ represents the parameters `weight2`, $Y$ represents the parameters `y`, $E_{\text{local}}$ represents `localMoeExpertNum = moeExpertNum / epWorldSize` (the number of routing MoE experts per Rank), $K$ represents the second dimension of `topkIds`.

    local variable description:
    - $\mathcal{T}_e$: Token index set routed to expert $e$, determined by `topkIds` after sorting.
    - $\hat{X}_e,\ S_{X,e}$: Quantized input for expert $e$ and its MX scaling factors, first stage intermediate results.
    - $W_{1,e}^{(G)}$, $W_{1,e}^{(U)}$: $W_1$ corresponds to the front $N/2$ column and back of the expert $e$ $N/2$ column submatrix is derived by splitting `weight1` according to gate branch and up branch.
    - $S_{1,e}^{(G)}$, $S_{1,e}^{(U)}$: MX scaling factors corresponding to $W_{1,e}^{(G)}$ and $W_{1,e}^{(U)}$, intercepted by dimension from `weightScales1`.
    - $S_{2,e}$: The MX scaling factor corresponding to $W_{2,e}$ comes from parameter `weightScales2`.
    - $G_e,\ U_e$: GMM1 gate branch and up branch output, intermediate results.
    - $A_e$: SwiGLU activation output, dimension $m_e \times N/2$, intermediate result.
    - $\hat{A}_e,\ S_{A,e}$: Quantized SwiGLU output and its MX scaling factors, intermediate results.
    - $O_e$: Expert output of GMM2, dimension $m_e \times H$, intermediate results.
    - $\pi(i, k)$: The row index of the $k$ top-k expert of Token $i$ after expansion sorting, determined by routing sorting.
    - $\mathrm{Q}_{\text{MX}}(\cdot)$: MX group-by-group quantization operation, block size = 32, output FP8 data and E8M0 scaling factor.
    - $\mathrm{DQ}_{\text{MX}}(\cdot)$: MX group-by-group dequantization operation, performed implicitly inside matmul.
    </details>

    <details>
    <summary> A8W4-FP quantization scene </summary>

    The first stage (Token selection, quantification and Dispatch):

    Input Token $X \in \mathbb{R}^{B\times H}$ for this rank. According to `topkIds`, get the Token subscript set $T_e$ corresponding to each expert $e$, and press 32 for the selected BF16 Token. A set of elements is quantized to MXFP8 E4M3:

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    Among them, the data type of $\hat{X}_e$ is `FLOAT8_E4M3FN`, and the data type of $S_{X,e}$ is `FLOAT8_E8M0`. The quantized Token and its scaling factor are then sent to the expert’s rank.

    Router MoE Expert's first and second layer weights $W_{1,e}$, $W_{2,e}$ are all MXFP4 E2M1 data with a scaling factor of E8M0, $e$ has a range of $[0,\text{localMoeExpertNum})$. The shared expert weights are provided individually by `sharedWeight1`, `sharedWeight2`, whose first dimension is `sharedExpertNumPerRank`. The A8W4 kernel handles FP4 weights in a matrix multiplication Prologue, and the logical data flow fed into the matrix multiplication is FP8 activations multiplied by FP4 weights.

    Second stage (GMM1, SwiGLU and re-quantification):

    For the token received by expert $e$, the first layer grouping matrix multiplication sum SwiGLU is calculated as:

    $$
    G_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(G)},S_{1,e}^{(G)}\right),
    $$

    $$
    U_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(U)},S_{1,e}^{(U)}\right),
    $$

    $$
    A_e = \operatorname{SwiGLU}(G_e,U_e).
    $$

    For the calculation method of SwiGLU, see [Activation formulas](#activation-formulas). Its output continues to be quantized into MXFP8 E4M3 in groups of 32 elements for use in the second layer of matrix multiplication:

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    third stage (GMM2):

    The second layer matrix multiplication is still A8W4, that is, FP8 E4M3 activation multiplies FP4 E2M1 weight:

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    When the shared expert is enabled (`sharedExpertNumPerRank` > 0), the shared expert locally performs the same GMM1 + SwiGLU + GMM2 calculation as the routing expert on each card for all tokens of the card, using `sharedWeight1`, `sharedWeight2`, `sharedWeightScales1`, `sharedWeightScales2` do not need to participate in Dispatch communication. The output of each shared expert is recorded as $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{sharedExpertNumPerRank} - 1\}$.

    The fourth stage (Combine and weighted merger):

    returns the output of each expert to the original rank, performs weighted merging according to `topkWeights`, and superimposes the shared expert output:

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{sharedExpertNumPerRank}-1} O^{\mathrm{shared}}_s[i],
    $$

    When shared experts are not enabled, the shared expert summation term is zero. The data type of the final output $Y$ is BF16. The main data type stream of A8W4-FP is: `BF16 -> MXFP8 E4M3 -> A8W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`.
    </details>

    <details>
    <summary> A4W4-FP quantization scene </summary>

    The first stage (Token selection, quantification and Dispatch):

    For the Token set $T_e$ corresponding to each expert $e$, the selected BF16 Token is quantized into MXFP4 E2M1 according to a set of 32 elements:

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    Among them, the data type of $\hat{X}_e$ is `FLOAT4_E2M1`, and the data type of $S_{X,e}$ is `FLOAT8_E8M0`. The first and second layer weights of the routing MoE experts $W_{1,e}$ and $W_{2,e}$ are both MXFP4 E2M1, and the range of $e$ is $[0,\text{localMoeExpertNum})$; the first dimension of the shared expert weight is `sharedExpertNumPerRank`. The weight scaling factor is E8M0.

    Phase 2 (A4W4 GMM1, SwiGLU and output type improvements):

    The first layer grouping matrix is multiplied by A4W4:

    $$
    G_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(G)},S_{1,e}^{(G)}\right),
    $$

    $$
    U_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(U)},S_{1,e}^{(U)}\right),
    $$

    $$
    A_e = \operatorname{SwiGLU}(G_e,U_e).
    $$

    For the calculation method of SwiGLU, see [Activation formulas](#activation-formulas). It is not possible to continue quantizing the SwiGLU output to FP4 here. When the kernel is at `QuantMode == E2M1_QUANT`, it specifies `SwigluQuantOutType` as `fp8_e4m3fn_t`, so the output is promoted to MXFP8 E4M3:

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    Stage 3 (A8W4 GMM2):

    Since the SwiGLU quantization output is FP8 E4M3 and the second layer weight is still FP4 E2M1, the second layer matrix multiplication is actually A8W4 instead of A4W4:

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    When the shared expert is enabled (`sharedExpertNumPerRank` > 0), the shared expert locally performs the same GMM1 + SwiGLU + GMM2 calculation as the routing expert on each card for all tokens of the card, using `sharedWeight1`, `sharedWeight2`, `sharedWeightScales1`, `sharedWeightScales2` do not need to participate in Dispatch communication. The output of each shared expert is recorded as $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{sharedExpertNumPerRank} - 1\}$.

    The fourth stage (Combine and weighted merger):

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{sharedExpertNumPerRank}-1} O^{\mathrm{shared}}_s[i],
    $$

    When shared experts are not enabled, the shared expert summation term is zero. The data type of the final output $Y$ is BF16. The complete data type stream of A4W4-FP is: `BF16 -> MXFP4 E2M1 -> A4W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`. All MX scaling factors are of type `FLOAT8_E8M0` and have a quantization granularity of 32 consecutive elements.
    </details>

## Parameter description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th> Parameter name </th>
   <th>Input/Output/Properties</th>
   <th>Description</th>
   <th> data type </th>
   <th> data format </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td> input </td>
   <td> This card communication domain information data. </td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td> input </td>
   <td>MoE layer input token hidden state. </td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkIds</td>
   <td> input </td>
   <td> expert index matrix, representing the topK experts selected by each token. The element value range is [0, moeExpertNum), and the topK experts selected by the same token cannot be repeated. </td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkWeights</td>
   <td> input </td>
   <td> indicates that the expert gating network of the MoE model is the gating weight coefficient corresponding to the topK experts selected by the current input token. </td>
   <td>FP32, BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight1</td>
   <td> input </td>
   <td>MoE The weight matrix (including gating and upward projection) of the first linear layer of the expert network is used to map the input to the intermediate dimension, and the output is supplied to the activation function. The routing MoE expert number is <code>localMoeExpertNum</code>. </td>
   <td>BF16, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td> input </td>
   <td>MoE The weight matrix of the second linear layer of the expert network is responsible for projecting the activated intermediate features back to the hidden dimension. The data type is consistent with weight1. The routing MoE expert number is <code>localMoeExpertNum</code>. </td>
   <td>BF16, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weightScales1</td>
   <td> Optional input </td>
   <td>MoE The quantified scaling factor of the weight matrix of the first linear layer of the expert network. </td>
   <td>FLOAT8_E8M0, UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScales2</td>
   <td> Optional input </td>
   <td>MoE The quantified scaling factor of the weight matrix of the second linear layer of the expert network. </td>
   <td>FLOAT8_E8M0, UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias1</td>
   <td> Optional input </td>
   <td>MoE expert network bias of the first linear layer. This parameter is only required in the A8W4-INT quantization scenario for accuracy compensation. </td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias2</td>
   <td> Optional input </td>
   <td>MoE The bias of the second linear layer of the expert network. This parameter is only needed in the A8W4-INT quantization scenario for accuracy compensation. </td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMask</td>
   <td> Optional input </td>
   <td> indicates whether the token participates in communication. </td>
   <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td> Optional input </td>
   <td> quantized smoothing parameter. </td>
   <td>FLOAT8_E8M0, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedWeight1</td>
   <td> Optional input </td>
   <td> shares the weight matrix (including gating and upward projection) of the first linear layer of the expert network, which is used to map the input to the intermediate dimension, and the output is supplied to the activation function. </td>
   <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeight2</td>
   <td> Optional input </td>
   <td> shares the weight matrix of the second linear layer of the expert network, which is responsible for projecting the activated intermediate features back to the hidden dimension. The data type is consistent with weight1. </td>
   <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeightScales1</td>
   <td> Optional input </td>
   <td> shares the quantified scaling factor of the weight matrix of the first linear layer of the expert network. </td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedWeightScales2</td>
   <td> Optional input </td>
   <td> Quantitative scaling factor of the weight matrix of the second linear layer of the shared expert network. </td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias1</td>
   <td> Optional input </td>
   <td> shares the bias of the first linear layer of the expert network, which is not supported yet. </td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias2</td>
   <td> Optional input </td>
   <td> shares the bias of the second linear layer of the expert network, which is not supported yet. </td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>Properties</td>
   <td>MoE model. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>Properties</td>
   <td> Expert parallel communication domain size. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>cclBufferSize</td>
   <td>Properties</td>
   <td>CCL communication buffer size. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>maxRecvTokenNum</td>
   <td> Optional attributes </td>
   <td> The maximum number of tokens each Rank can receive. The default value is 0. When the value is 0, the memory size will be reserved according to the maximum value bs*epWorldSize*min(topK, localMoeExpertNum); when it is not 0, the memory will be reserved according to the input value. In this scenario, the user needs to ensure that the filled-in value is greater than or equal to the maximum number of tokens that can be received by each rank. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantMode</td>
   <td> Optional attributes </td>
   <td>dispatch communication quantization mode. 0 represents non-quantization (A16W16 scene), 2 represents INT8 quantization (A8W8-INT, A8W4-INT scenes), and 4 represents MXFP quantization (A8W8-FP, A8W4-FP, A4W4-FP scenes). The default value is 0. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantOutDtype</td>
   <td> Optional attributes </td>
   <td>dispatch The data type output after quantization. Supports 1 (INT8), 23 (FLOAT8_E5M2), 24 (FLOAT8_E4M3FN), 296 (FLOAT4_E2M1). The default value is DT_UNDEFINED. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>combineQuantMode</td>
   <td> Optional attributes </td>
   <td>combine quantization mode during communication. 0 represents non-quantization, 3 represents MXFP float8_e5m2 type, 4 represents MXFP float8_e4m3 type, and the default value is 0. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td> Optional attributes </td>
   <td> is a reserved parameter and is not supported yet. The default value is "". </td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>numMaxTokensPerRank</td>
   <td> Optional attributes </td>
   <td> The number of tokens on each card. When the numTokens of each rank are different, it is the maximum numTokens size. The default value is 0. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>activation</td>
   <td> Optional attributes </td>
   <td> activation function type. The default value is "swiglu". Optional values ​​are "swiglu", "swiglustep", "swigluoai" and "situglu". </td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>activation_params</td>
   <td> Optional attributes </td>
   <td> activation function parameter list, the default value is []. The order and number of parameters are determined by activation: "swiglu" and "swiglustep" support [] or [clamp]; "swigluoai" supports [] or [clamp, alpha, beta]; "situglu" supports [], [beta] or [beta, linear_beta]. When using an empty list, clamp defaults to the maximum value of float, alpha defaults to 1.702, beta defaults to 1.0, and linear_beta is not enabled. clamp needs to be ≥ 0 and cannot be NaN; the beta of alpha and swigluoai need to be finite values; the beta and linear_beta of situglu, as divisors, need to be finite non-zero values. </td>
   <td>LIST_FLOAT</td>
   <td></td>
  </tr>
  <tr>
   <td>activationOutDtype</td>
   <td> Optional attributes </td>
   <td> The data type output by the activation function. The default value is DT_UNDEFINED. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight1</td>
   <td> Optional attributes </td>
   <td>weight1 is transposed. The default value is false. </td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight2</td>
   <td> Optional attributes </td>
   <td>weight2 is transposed. The default value is false. </td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>weight1Interleave</td>
   <td> Optional attributes </td>
   <td>weight1 staggered parameters. Reserved parameter, default value is 0. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>topoType</td>
   <td> Optional attributes </td>
   <td> communication topology type, automatically derived from the communication domain context. 0 indicates MTE topology, and 1 indicates URMA spanning super topology. Currently, the URMA communication method is not supported. The default value is 0. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>rankNumPerServer</td>
   <td> Optional attributes </td>
   <td> The number of ranks on each server is at least 2. The default value is 2. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>topkWeightsType</td>
   <td> Optional attributes </td>
   <td>topkWeights forward switch. 0 means off, 1 means on (topkWeights will be sent to the target rank in advance along with the token data in the dispatch phase to reduce the communication volume in the combine phase). The default value is 0. </td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>y</td>
   <td> output </td>
   <td> calculates the output result, the data type is the same as the input x. </td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNums</td>
   <td> output </td>
   <td> The actual number of tokens received by each expert on this card. </td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## Constraint description

- **Reserved and non-external parameter description**:
  - Some parameters and data types in the parameter table are not publicly available yet and are reserved or used for internal implementation. The interface parameters and their constraints are detailed in the [MegaMoE operator interface document](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md).

- **Parameter consistency constraints**:
  - The values of parameters such as `moeExpertNum`, `maxRecvTokenNum`, `dispatchQuantMode`, `dispatchQuantOutDtype`, and `numMaxTokensPerRank` used in the process of calling the operator must be consistent across all cards and across different layers in the network.

- **Communication domain and networking constraints**:
  - The values of `epWorldSize` and `cclBufferSize` parameters of all cards must be consistent.
  - The driver version of each node in the communication domain should be the same.
  - <term>Atlas A2 training series products/Atlas A2 inference series products </term>: Multi-machine communication domain requires switch networking and does not support dual-machine direct connection networking.
  - <term>Atlas A3 training series product/Atlas A3 inference series product </term>: The multi-machine communication domain is required to be within a super node, and dual-machine direct connection networking and cross-super node networking are not supported.
  - <term>Ascend 950PR/Ascend 950DT</term>: Only supports UB Memory communication protocol.
- **Parameter constraints**:
  - **<term>Atlas A2 training series products/Atlas A2 inference series products </term>, <term>Atlas A3 training series products/Atlas A3 inference series products </term>**:
    - Scene matching matrix:

      | scene | x | weight1 | weight2 | weightScales1 | weightScales2 | bias1 | bias2 | y | dispatchQuantMode | dispatchQuantOutDtype |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | A16W16 | BF16 | BF16 | BF16 | – | – | – | – | BF16 | 0 | – |
      | A8W8-INT | BF16 | INT8 | INT8 | UINT64 | UINT64 | – | – | BF16 | 2 | 1 (INT8) |
      | A8W4-INT | BF16 | INT4(INT32) | INT4(INT32) | UINT64 | UINT64 | FP32 | FP32 | BF16 | 2 | 1 (INT8) |

  - **<term>Ascend 950PR/Ascend 950DT</term>**:
    - `activation` only supports "swiglu".
    - `activation_params` only supports [] or [clamp].
    - `BS` (`x`.dim0) supports [1, +∞), the actual upper limit is subject to `cclBufferSize`. The operator adopts a batch processing mechanism, the BS is no longer subject to the hard limit of UB capacity, and the dispatch stage is processed in batches according to a fixed granularity.
    - `H` (`x`.dim1) supports 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192.
    - `topK` (`topkIds`.dim1) supports [1, 32].
    - `expertPerRank` Range [1, 1024].
    - `hiddenDim` (`weight1`.dim1) only supports 1024, 2048, 3072, 4096, 7168.
    - `epWorldSize` range [2, 1024].
    - `moeExpertNum` range [`epWorldSize`, 2048], and `moeExpertNum` % `epWorldSize` == 0.
    - `maxRecvTokenNum` range [0, `BS` × `epWorldSize` × min(`topK`, `localMoeExpertNum`)].
    - `dispatchQuantOutDtype` only supports 23 (FLOAT8_E5M2) or 24 (FLOAT8_E4M3FN) or 296 (FLOAT4_E2M1).
    - The current version only supports MXFP quantization mode (`dispatchQuantMode` = 4), the dispatch stage uses MX group-by-group quantization (group size = 32), and the quantization scaling factor type is FLOAT8_E8M0.
    - `combineQuantMode` takes values 0, 3, 4, 0 represents non-quantization, 3 represents MXFP float8_e5m2 type, 4 represents MXFP float8_e4m3 type
    - `commAlg` must be the empty string "".
    - The data type of `y` is the same as `x`.
    - The dim1 of `weight1` (`hiddenDim`) must be equal to twice the dim2 of `weight2`. This is because SwiGLU activation needs to halve the intermediate dimension from `hiddenDim` to `hiddenDim`/2.
    - `localMoeExpertNum` = `moeExpertNum` / `epWorldSize`; `sharedExpertNumPerRank` = `sharedWeight1`.dim0 (0 when shared experts are not enabled); `expertPerRank` = `sharedExpertNumPerRank` + `localMoeExpertNum`.
    - `sharedExpertNumPerRank` range [0, 4].
    - `topoType` is automatically derived from the communication domain context. 0 indicates MTE topology, and 1 indicates URMA spanning super topology. Currently, the URMA communication method is not supported.
    - `topkWeightsType` takes the value 0 or 1, 0 means turning off topkWeights forward movement, 1 means turning it on. Currently, the URMA communication method is not supported.
    - `numMaxTokensPerRank` is automatically calculated when it is 0; when it is not 0, it must be equal to `BS`.
    - `cclBufferSize` requires >= full card soft synchronization reserved space (fixed 60KB) + mask receiving space + quantized token scaling factor space + combine sending space.
    - `weightScales1` and `weightScales2` are required inputs, and the data type must be FLOAT8_E8M0.
    - The data types of `weight1` and `weight2` must be consistent, and only support FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT4_E2M1.
    - `topkWeights` data type only supports BF16 or FP32.
    - The current versions of `xActiveMask` and `scales` do not support non-null input, and a null pointer needs to be passed in.

  - **MXFP quantized scene constraints**:
      - `weight1` shape is (`localMoeExpertNum`, `hiddenDim`, `H`), `weight2` The shape is (`localMoeExpertNum`, `H`, `hiddenDim`/2).
      - `weightScales1` shape is (`localMoeExpertNum`, `hiddenDim`, CeilDiv(`H`, 64), 2).
      - `weightScales2` shape is (`localMoeExpertNum`, `H`, CeilDiv(`hiddenDim`/2, 64), 2).
      - `sharedWeight1` shape is (`sharedExpertNumPerRank`, `hiddenDim`, `H`), `sharedWeight2` The shape is (`sharedExpertNumPerRank`, `H`, `hiddenDim`/2).
      - `sharedWeightScales1` shape is (`sharedExpertNumPerRank`, `hiddenDim`, CeilDiv(`H`, 64), 2), `sharedWeightScales2` The shape is (`sharedExpertNumPerRank`, `H`, CeilDiv(`hiddenDim`/2, 64), 2).
      - dim3 of `weightScales1` and dim3 of `weightScales2` must be equal to 2.
      - In the A8W4-FP scenario, the FLOAT4_E2M1 type `weight1` must use the FORMAT_FRACTAL_NZ_C0_32 format.
      - In the MXFP scenario, when `dispatchQuantOutDtype`=23, `weight1` and `weight2` must be FLOAT8_E5 M2, it must be FLOAT8_E4M3FN when `dispatchQuantOutDtype`=24, and it must be FLOAT4_E2M1 when `dispatchQuantOutDtype`=296.

## Calling instructions

| Calling method | Sample code | Description |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch interface call | - | Call the `mega_moe` operator through the [MegaMoE PyTorch interface](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md). |
