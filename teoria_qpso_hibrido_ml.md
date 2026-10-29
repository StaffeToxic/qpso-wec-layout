# Fundamentação Teórica e Matemática: QPSO Híbrido com Machine Learning Surrogate

## Aplicação à Otimização de Layouts de Wave Energy Converters (WECs)

---

## Sumário

1. [Introdução e Motivação](#1-introdução-e-motivação)
2. [Machine Learning Surrogate (Random Forest)](#2-machine-learning-surrogate-random-forest)
3. [Decomposição SHAP para Avaliação de Fitness](#3-decomposição-shap-para-avaliação-de-fitness)
4. [Particle Swarm Optimization (PSO) Clássico](#4-particle-swarm-optimization-pso-clássico)
5. [QPSO — Quantum-Inspired Particle Swarm Optimization](#5-qpso--quantum-inspired-particle-swarm-optimization)
6. [Discretização via Função Sigmoid](#6-discretização-via-função-sigmoid)
7. [Operador Genético Híbrido (Mutação GA)](#7-operador-genético-híbrido-mutação-ga)
8. [Formulação Completa do QPSO Híbrido Discreto](#8-formulação-completa-do-qpso-híbrido-discreto)
9. [Versão Enhanced: Multi-Estratégia e Operadores Avançados](#9-versão-enhanced-multi-estratégia-e-operadores-avançados)
10. [Função de Fitness Multi-Objetivo](#10-função-de-fitness-multi-objetivo)
11. [Análise de Convergência](#11-análise-de-convergência)
12. [Comparativo com Outros Algoritmos Híbridos (QAOA, QEA)](#12-comparativo-com-outros-algoritmos-híbridos-qaoa-qea)
13. [Referências](#13-referências)

---

## 1. Introdução e Motivação

### 1.1 O Problema de Otimização de Layouts de WECs

A otimização do posicionamento de **Wave Energy Converters (WECs)** é um problema combinatório NP-difícil. Dado um conjunto de $N$ boias distribuídas em um campo marítimo, deseja-se selecionar um subconjunto que maximize a potência total capturada enquanto minimiza custos operacionais.

**Formulação do problema:**

$$
\max_{\mathbf{b} \in \{0,1\}^N} \quad F(\mathbf{b}) = w_p \cdot P(\mathbf{b}) + w_e \cdot E(\mathbf{b})
$$

$$
\text{sujeito a} \quad \sum_{i=1}^{N} b_i \geq N_{\min}
$$

onde:
- $\mathbf{b} = (b_1, b_2, \ldots, b_N) \in \{0,1\}^N$ é o vetor de decisão binário
- $b_i = 1$ indica que a boia $i$ está **ativa** (mantida no layout)
- $b_i = 0$ indica que a boia $i$ está **desligada** (removida)
- $P(\mathbf{b})$ é a potência total predita pelo modelo ML
- $E(\mathbf{b}) = P(\mathbf{b}) / \sum b_i$ é a eficiência por boia
- $w_p, w_e$ são pesos da função objetivo ($w_p = 0.7$, $w_e = 0.3$)
- $N_{\min}$ é o número mínimo de boias ativas (restrição de viabilidade)

### 1.2 Justificativa da Abordagem Híbrida

O espaço de busca é de $2^{49} \approx 5.63 \times 10^{14}$ configurações possíveis. A busca exaustiva é computacionalmente inviável. A abordagem proposta combina três paradigmas:

| Componente | Papel | Inspiração |
|:---|:---|:---|
| **ML Surrogate** | Substituir simulação BEM cara (~horas) | Inteligência Artificial |
| **QPSO** | Exploração do espaço de busca via mecânica quântica | Mecânica Quântica |
| **Mutação GA** | Escape de ótimos locais | Computação Evolutiva |

---

## 2. Machine Learning Surrogate (Random Forest)

### 2.1 Motivação

A simulação numérica via **Boundary Element Method (BEM)** para calcular a potência de um layout de WECs é extremamente custosa computacionalmente ($\mathcal{O}(\text{horas})$ por configuração). Um **modelo surrogate** (substituto) treinado a partir de simulações BEM pré-existentes permite avaliar novas configurações em $\mathcal{O}(\text{milissegundos})$.

### 2.2 Random Forest Regressor

O **Random Forest** (Breiman, 2001) é um ensemble de $T$ árvores de regressão $\{h_1, h_2, \ldots, h_T\}$, onde cada árvore é treinada sobre uma subamostra bootstrap dos dados com seleção aleatória de features em cada nó.

**Predição:**

$$
\hat{f}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})
$$

onde $\mathbf{x} \in \mathbb{R}^{2N}$ é o vetor de coordenadas $(X_1, Y_1, X_2, Y_2, \ldots, X_N, Y_N)$.

**Features de entrada:**

$$
\mathbf{x} = (X_1, Y_1, X_2, Y_2, \ldots, X_{49}, Y_{49}) \in \mathbb{R}^{98}
$$

**Target:**

$$
y = P_{\text{total}} \quad \text{(Potência total em Watts)}
$$

### 2.3 Treinamento e Validação

O dataset é dividido em treino (80%) e teste (20%) com seed fixa:

$$
\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{test}}, \quad |\mathcal{D}_{\text{test}}| = 0.2 \cdot |\mathcal{D}|
$$

**Métricas de validação:**

- **R² Score (Coeficiente de Determinação):**

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

- **MAE (Erro Médio Absoluto):**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Um $R^2 > 0.85$ indica que o modelo captura mais de 85% da variância nos dados reais, sendo suficiente para uso como surrogate.

### 2.4 Hiperparâmetros do Modelo

| Parâmetro | Valor | Justificativa |
|:---|:---|:---|
| `n_estimators` | 150–200 | Número de árvores no ensemble |
| `max_depth` | `None` | Captura interações complexas entre boias |
| `min_samples_split` | 2 (default sklearn) | Controle de overfitting |
| `n_jobs` | -1 | Paralelismo total na CPU |
| `random_state` | 42 | Reprodutibilidade |

---

## 3. Decomposição SHAP para Avaliação de Fitness

### 3.1 O Problema Out-of-Distribution (OOD)

Quando uma boia $i$ é "desligada", a abordagem ingênua de zerar suas coordenadas ($X_i = 0, Y_i = 0$) gera um ponto **fora da distribuição de treinamento** (OOD). O modelo Random Forest não foi treinado com boias em $(0, 0)$, resultando em predições não-confiáveis.

### 3.2 SHAP — SHapley Additive exPlanations

A teoria de **Shapley** (Shapley, 1953) da teoria dos jogos cooperativos define o valor justo de cada jogador em uma coalizão. Aplicado ao ML, o **valor SHAP** de uma feature $i$ quantifica sua contribuição marginal para a predição:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]
$$

onde:
- $N$ é o conjunto de todas as features
- $S$ é um subconjunto de features (sem a feature $i$)
- $f(S)$ é a predição do modelo usando apenas as features em $S$

### 3.3 TreeExplainer (Exato para Random Forests)

Para modelos baseados em árvore, o **TreeExplainer** (Lundberg et al., 2020) calcula os valores SHAP de forma **exata** em tempo polinomial $\mathcal{O}(TLD^2)$, onde $T$ é o número de árvores, $L$ é o número de folhas e $D$ é a profundidade.

**Propriedade aditiva fundamental:**

$$
\hat{f}(\mathbf{x}) = \phi_0 + \sum_{j=1}^{2N} \phi_j(\mathbf{x})
$$

onde $\phi_0 = \mathbb{E}[\hat{f}(\mathbf{X})]$ é o valor base (média global do modelo).

### 3.4 Agrupamento por WEC

Cada boia $i$ possui duas features: $X_i$ e $Y_i$. A contribuição total da boia $i$ é:

$$
c_i = \phi_{X_i} + \phi_{Y_i}
$$

resultando em um vetor de contribuições por boia:

$$
\mathbf{c} = (c_1, c_2, \ldots, c_N) \in \mathbb{R}^N
$$

### 3.5 Avaliação de Fitness via SHAP

Quando um subconjunto de boias $\mathcal{A} = \{i : b_i = 1\}$ está ativo, a potência estimada é:

$$
\hat{P}(\mathbf{b}) = \phi_0 + \sum_{i \in \mathcal{A}} c_i = \phi_0 + \sum_{i=1}^{N} b_i \cdot c_i
$$

**Vantagem:** Esta fórmula é **linear e aditiva**, evitando qualquer predição OOD. Remover uma boia simplesmente subtrai sua contribuição SHAP.

**Verificação de integridade (sanity check):**

$$
\left| \phi_0 + \sum_{i=1}^{N} c_i - \hat{f}(\mathbf{x}_{\text{ref}}) \right| < \epsilon
$$

onde $\epsilon$ é um limiar de tolerância numérica e $\mathbf{x}_{\text{ref}}$ é o layout de referência.

---

## 4. Particle Swarm Optimization (PSO) Clássico

### 4.1 Formulação Original

O PSO (Kennedy & Eberhart, 1995) modela um enxame de $M$ partículas em um espaço contínuo $\mathbb{R}^D$. Cada partícula $i$ possui:

- **Posição:** $\mathbf{x}_i^{(t)} \in \mathbb{R}^D$
- **Velocidade:** $\mathbf{v}_i^{(t)} \in \mathbb{R}^D$
- **Melhor pessoal:** $\mathbf{p}_i = \arg\max_{\tau \leq t} f(\mathbf{x}_i^{(\tau)})$
- **Melhor global:** $\mathbf{g} = \arg\max_{i,\tau} f(\mathbf{x}_i^{(\tau)})$

**Equações de atualização do PSO clássico:**

$$
\mathbf{v}_i^{(t+1)} = \omega \mathbf{v}_i^{(t)} + c_1 r_1 \left(\mathbf{p}_i - \mathbf{x}_i^{(t)}\right) + c_2 r_2 \left(\mathbf{g} - \mathbf{x}_i^{(t)}\right)
$$

$$
\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}
$$

onde $\omega$ é o peso inercial, $c_1, c_2$ são coeficientes de aceleração, e $r_1, r_2 \sim U(0,1)$.

### 4.2 Limitações do PSO Clássico

1. **Convergência prematura** (fica preso em ótimos locais)
2. **Depende do vetor de velocidade** (parâmetro extra a calibrar)
3. **Não naturalmente discreto** (necessita adaptação para binário)

Estas limitações motivam a transição para o QPSO.

---

## 5. QPSO — Quantum-Inspired Particle Swarm Optimization

### 5.1 Motivação Quântica

O QPSO (Sun, Feng & Xu, 2004) substitui a trajetória determinística do PSO por um modelo probabilístico inspirado na **mecânica quântica**. Em vez de os movimentos das partículas seguirem trajetórias Newtonianas, cada partícula é modelada como estando dentro de um **poço de potencial delta de Dirac**, onde sua posição é descrita por uma **função de onda** $\psi(\mathbf{x})$.

### 5.2 Equação de Schrödinger Estacionária

Uma partícula em um poço de potencial delta de Dirac unidimensional centrado em $p$ satisfaz a equação de Schrödinger independente do tempo:

$$
-\frac{\hbar^2}{2m} \frac{d^2\psi}{dx^2} + V(x)\psi(x) = E\psi(x)
$$

com potencial atrativo:

$$
V(x) = -\gamma \delta(x - p)
$$

onde $\delta$ é a função delta de Dirac, $\gamma > 0$ é a intensidade do poço, e $p$ é o centro (ponto atrator).

### 5.3 Solução da Função de Onda

A solução normalizada para o estado fundamental ($E < 0$) é:

$$
\psi(x) = \frac{1}{\sqrt{L}} \exp\left(-\frac{|x - p|}{L}\right)
$$

onde $L = \hbar^2 / (m\gamma)$ é o **comprimento característico** do confinamento.

### 5.4 Densidade de Probabilidade

A densidade de probabilidade de encontrar a partícula na posição $x$ é:

$$
|\psi(x)|^2 = \frac{1}{L} \exp\left(-\frac{2|x - p|}{L}\right)
$$

Isto é uma **distribuição de Laplace** (exponencial dupla) centrada em $p$.

### 5.5 Amostragem por Inversão da CDF

Para amostrar da distribuição $|\psi(x)|^2$ via **inverse transform sampling**, calcula-se a CDF e inverte-se:

Seja $u \sim U(0, 1)$. Então:

$$
x = p \pm \frac{L}{2} \ln\left(\frac{1}{u}\right)
$$

onde o sinal $\pm$ é escolhido com probabilidade $0.5$ cada.

Esta é a **equação central do QPSO**: a partícula "tunela" quanticamente para uma nova posição, em vez de mover-se por uma trajetória determinística.

### 5.6 Ponto Atrator Estocástico

O centro do poço de potencial $p_i$ para cada partícula $i$ é uma combinação convexa estocástica entre o melhor pessoal ($\mathbf{p}_i$) e o melhor global ($\mathbf{g}$):

$$
p_{i,j}^{(t)} = \varphi_{i,j} \cdot p_{i,j} + (1 - \varphi_{i,j}) \cdot g_j, \quad \varphi_{i,j} \sim U(0, 1)
$$

para cada dimensão $j \in \{1, \ldots, D\}$.

### 5.7 Mean Best Position (mbest)

O **mbest** é a média das melhores posições pessoais de todas as partículas, funcionando como um **centroide** do enxame:

$$
\overline{\mathbf{p}}^{(t)} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{p}_i^{(t)}
$$

No caso discreto (binário), cada componente de mbest é a **frequência** do bit ativo:

$$
\overline{p}_j = \frac{1}{M} \sum_{i=1}^{M} p_{i,j}
$$

### 5.8 Comprimento Característico $L$

O comprimento $L$ controla a dispersão da distribuição de Laplace. Para que a amostragem da §5.5 — $x = p \pm \tfrac{L}{2}\ln(1/u)$ — reproduza a forma canônica da §5.9, o comprimento característico **inclui um fator 2**:

$$
L_{i,j}^{(t)} = 2\,\alpha^{(t)} \cdot \left| \overline{p}_j^{(t)} - x_{i,j}^{(t)} \right|
$$

> **Nota (convenção do fator 2 — corrige inconsistência da dedução).** Substituindo este $L$ em §5.5 obtém-se exatamente $x = p \pm \alpha^{(t)}\,\big|\overline{p}_j - x_{i,j}\big|\,\ln(1/u)$ (§5.9). A versão anterior, $L = \alpha\,|\cdot|$ (sem o 2), era inconsistente: levada a §5.5 produziria **metade** do passo canônico. Na literatura de QPSO o coeficiente de contração-expansão $\beta \equiv \alpha$ já absorve esse fator 2, e a implementação aplica diretamente a forma da §5.9 (sem o $\tfrac{1}{2}$) — portanto o **código está correto**; o ajuste aqui é apenas na dedução do texto. (Verificação numérica: com a amostragem de §5.5 o passo médio é $L/2$.)

onde $\alpha^{(t)}$ é o **coeficiente de contração-expansão** que decai linearmente:

$$
\alpha^{(t)} = \alpha_{\max} - \left(\alpha_{\max} - \alpha_{\min}\right) \cdot \frac{t}{T}
$$

Tipicamente, $\alpha_{\max} = 1.0$ e $\alpha_{\min} = 0.5$, resultando em:

$$
\alpha^{(t)} = 1.0 - 0.5 \cdot \frac{t}{T}
$$

- **$\alpha$ alto** (início): partículas exploram amplamente (distribuição larga)
- **$\alpha$ baixo** (fim): partículas convergem e refinam (distribuição estreita)

### 5.9 Equação de Atualização Completa do QPSO

Reunindo todos os componentes, a posição contínua de cada partícula é atualizada por:

$$
\boxed{x_{i,j}^{(t+1)} = p_{i,j}^{(t)} \pm \alpha^{(t)} \cdot \left|\overline{p}_j^{(t)} - x_{i,j}^{(t)}\right| \cdot \ln\left(\frac{1}{u_{i,j}}\right)}
$$

onde:
- $p_{i,j}^{(t)} = \varphi_{i,j} \cdot \text{pbest}_{i,j} + (1 - \varphi_{i,j}) \cdot \text{gbest}_j$
- $\varphi_{i,j} \sim U(0, 1)$
- $u_{i,j} \sim U(\epsilon, 1)$ com $\epsilon = 10^{-10}$ para estabilidade numérica
- O sinal $\pm$ é escolhido aleatoriamente com $P(+) = P(-) = 0.5$

### 5.10 Vantagens sobre o PSO Clássico

| Propriedade | PSO Clássico | QPSO |
|:---|:---|:---|
| Parâmetro velocidade | Necessário ($\omega, c_1, c_2$) | **Eliminado** |
| Tipo de movimento | Trajetória determinística | **Tunneling probabilístico** |
| Escape de ótimos locais | Fraco | Forte (distribuição Laplace) |
| Convergência teórica | Não garantida globalmente | **Favorecida** (Sun et al., 2004; ver §11.1) |

---

## 6. Discretização via Função Sigmoid

### 6.1 Necessidade de Discretização

O QPSO opera naturalmente no domínio contínuo $\mathbb{R}^D$. Para o problema de seleção de WECs, precisamos de decisões binárias $\{0, 1\}$. A **função sigmoid** (logística) mapeia o valor real para uma probabilidade de ativação:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### 6.2 Propriedades da Sigmoid

- $\sigma: \mathbb{R} \to (0, 1)$ — sempre retorna uma probabilidade válida
- $\sigma(0) = 0.5$ — ponto de decisão neutra
- $\lim_{x \to +\infty} \sigma(x) = 1$ — certeza de ativação
- $\lim_{x \to -\infty} \sigma(x) = 0$ — certeza de desativação
- $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ — gradiente suave

### 6.3 Clipping Numérico

Para evitar overflow/underflow na exponencial:

$$
\sigma_{\text{clip}}(x) = \sigma\left(\text{clip}(x, -10, 10)\right)
$$

Com clipping adicional na probabilidade:

$$
\text{prob}_{i,j} = \text{clip}\left(\sigma(x_{i,j}^{(t+1)}), 0.01, 0.99\right)
$$

Isso garante que nenhuma dimensão se torne deterministicamente 0 ou 1, preservando a capacidade exploratória.

### 6.4 Amostragem Estocástica (Colapso da Função de Onda)

A decisão binária é tomada por amostragem de Bernoulli:

$$
b_{i,j}^{(t+1)} = \begin{cases} 1 & \text{se } r_{i,j} < \text{prob}_{i,j} \\ 0 & \text{caso contrário} \end{cases}, \quad r_{i,j} \sim U(0, 1)
$$

Esta operação é análoga ao **colapso da função de onda** em mecânica quântica: a posição contínua (superposição) "colapsa" para um estado discreto (observável) mediante amostragem.

### 6.5 Interpretação Física

| Conceito Quântico | Análogo no QPSO Discreto |
|:---|:---|
| Função de onda $\psi(x)$ | Posição contínua $x_{i,j}^{(t)}$ |
| $|\psi|^2$ (probabilidade) | $\sigma(x_{i,j})$ |
| Colapso / Medição | Amostragem Bernoulli $r < \sigma(x)$ |
| Superposição $|0\rangle + |1\rangle$ | Valor contínuo entre 0 e 1 |
| Estado observado $|0\rangle$ ou $|1\rangle$ | Bit $b_{i,j} \in \{0, 1\}$ |

---

## 7. Operador Genético Híbrido (Mutação GA)

### 7.1 Motivação

Apesar do tunneling quântico do QPSO melhorar a exploração, a convergência para regiões específicas do espaço pode levar a estagnação em **ótimos locais**. O operador de **mutação genética** (inspirado em Algoritmos Genéticos) introduz perturbações aleatórias que permitem escapar dessas armadilhas.

### 7.2 Formulação da Mutação Seletiva

A mutação é aplicada **seletivamente** apenas às partículas com fitness abaixo da mediana:

**Algoritmo:**

1. Calcular a mediana do fitness das partículas válidas:
$$
\tilde{f} = \text{median}\left(\{f_i : f_i > -10^5\}\right)
$$

2. Para cada partícula $i$ com $f_i < \tilde{f}$:
    - Com probabilidade $p_{\text{mut}}$, inverter $n$ bits aleatórios:
$$
b_{i,k} \leftarrow 1 - b_{i,k}, \quad k \in \text{RandomChoice}(\{1,\ldots,N\}, n)
$$

3. Reparar restrições: se $\sum_j b_{i,j} < N_{\min}$, ativar bits aleatórios até satisfazer.

### 7.3 Número de Bits Invertidos (Multi-bit Flip)

Para melhorar a exploração, o número de bits invertidos é proporcional à taxa de mutação e à dimensão:

$$
n_{\text{flips}} = \max\left(1, \left\lfloor p_{\text{mut}} \cdot N \cdot \rho \right\rfloor\right)
$$

onde $\rho$ é um fator de intensidade da mutação multi-bit. **Nota de implementação:** a versão legada (`qpso_ml_standalone.py`) usa $\rho = 0{,}3$ (mutação mais agressiva); a versão refatorada (`qpso_discreto.py`) usa $\rho = 0{,}1$. Com $p_{\text{mut}} = 0{,}35$ e $N = 49$, isso dá $n_{\text{flips}} = 5$ (legada) vs $n_{\text{flips}} = 1$ (refatorada) no início da otimização.

### 7.4 Mutação Adaptativa (Versão Enhanced)

Na versão melhorada, a taxa de mutação decai linearmente ao longo das iterações:

$$
p_{\text{mut}}(t) = p_{\text{base}} + (p_{\text{final}} - p_{\text{base}}) \cdot \frac{t}{T}
$$

Na implementação: $p_{\text{base}} = 0.35$ (início: alta exploração), $p_{\text{final}} = 0.05$ (fim: refinamento).

$$
\boxed{p_{\text{mut}}(t) = 0.35 - 0.30 \cdot \frac{t}{T}}
$$

**Justificativa:** No início da otimização, perturbações agressivas ajudam a explorar o espaço. Nas iterações finais, perturbações sutis refinam as melhores soluções sem destruí-las.

---

## 8. Formulação Completa do QPSO Híbrido Discreto

### 8.1 Algoritmo Passo a Passo

**Entrada:**
- $N$ = número de WECs (49)
- $M$ = número de partículas (25–30)
- $T$ = número máximo de iterações (60–80)
- $N_{\min}$ = mínimo de boias ativas (20–25)
- $\hat{f}$ = modelo ML surrogate treinado
- $\mathbf{x}_{\text{ref}}$ = layout de referência (melhor do dataset)

**Inicialização ($t = 0$):**

1. Gerar população binária: $\mathbf{b}_i^{(0)} \in \{0,1\}^N$ com $\sim 70\%$ de bits ativos
2. Definir partícula $\mathbf{b}_0^{(0)} = \mathbf{1}$ (todas ativas) como referência
3. Inicializar $\text{pbest}_i = \mathbf{b}_i^{(0)}$, $\text{pbest\_fit}_i = -\infty$, $\text{gbest\_fit} = -\infty$

**Loop principal ($t = 1, 2, \ldots, T$):**

---

**Passo 1 — Avaliação de Fitness:**

Para cada partícula $i$:

$$
f_i = F(\mathbf{b}_i^{(t)}) = w_p \cdot \hat{P}(\mathbf{b}_i^{(t)}) + w_e \cdot \frac{\hat{P}(\mathbf{b}_i^{(t)})}{\sum_j b_{i,j}^{(t)}}
$$

Atualizar:
$$
\text{pbest}_i \leftarrow \begin{cases} \mathbf{b}_i^{(t)} & \text{se } f_i > \text{pbest\_fit}_i \\ \text{pbest}_i & \text{caso contrário} \end{cases}
$$
$$
\text{gbest} \leftarrow \arg\max\limits_i \text{pbest\_fit}_i
$$

---

**Passo 2 — Calcular mbest e $\alpha$:**

$$
\overline{p}_j = \frac{1}{M} \sum_{i=1}^{M} \text{pbest}_{i,j}, \quad j = 1, \ldots, N
$$

$$
\alpha^{(t)} = 1.0 - 0.5 \cdot \frac{t}{T}
$$

---

**Passo 3 — Atualização Quântica (Schrödinger + Sigmoid + Colapso):**

Para cada partícula $i$, para cada dimensão $j$:

$$
\varphi_{i,j} \sim U(0,1)
$$

$$
p_{i,j} = \varphi_{i,j} \cdot \text{pbest}_{i,j} + (1 - \varphi_{i,j}) \cdot \text{gbest}_j
$$

$$
u_{i,j} \sim U(10^{-10}, 1)
$$

$$
L_{i,j} = \alpha^{(t)} \cdot \left| \overline{p}_j - b_{i,j}^{(t)} \right|
$$

$$
d_{i,j} = \begin{cases} +1 & \text{com prob. } 0.5 \\ -1 & \text{com prob. } 0.5 \end{cases}
$$

$$
x_{i,j}^{(t+1)} = p_{i,j} + d_{i,j} \cdot L_{i,j} \cdot \ln\left(\frac{1}{u_{i,j}}\right)
$$

$$
\text{prob}_{i,j} = \sigma\left(x_{i,j}^{(t+1)}\right) = \frac{1}{1 + e^{-x_{i,j}^{(t+1)}}}
$$

$$
b_{i,j}^{(t+1)} = \begin{cases} 1 & \text{se } r_{i,j} < \text{prob}_{i,j}, \quad r_{i,j} \sim U(0,1) \\ 0 & \text{caso contrário} \end{cases}
$$

---

**Passo 4 — Mutação Genética Híbrida:**

$$
\tilde{f} = \text{median}(\{f_i : f_i > -10^5\})
$$

Para cada $i$ com $f_i < \tilde{f}$: com probabilidade $p_{\text{mut}}$, inverter $n_{\text{flips}}$ bits aleatórios.

---

**Passo 5 — Reparação de Restrições:**

$$
\text{Se } \sum_j b_{i,j}^{(t+1)} < N_{\min}: \text{ ativar bits aleatórios até } \sum_j b_{i,j} = N_{\min}
$$

---

### 8.2 Diagrama de Fluxo

```
┌─────────────────────────────────────────────────────────┐
│              QPSO HÍBRIDO DISCRETO + ML                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  ETAPA 1: Treinar Random Forest Surrogate    │       │
│  │  Dados BEM → RF → Predição rápida            │       │
│  └──────────────────┬───────────────────────────┘       │
│                     │                                   │
│  ┌──────────────────▼───────────────────────────┐       │
│  │  ETAPA 2: Calcular SHAP Contributions         │       │
│  │  RF + TreeExplainer → c_i para cada WEC       │       │
│  └──────────────────┬───────────────────────────┘       │
│                     │                                   │
│  ┌──────────────────▼───────────────────────────┐       │
│  │  ETAPA 3: Inicializar Enxame Binário          │       │
│  │  M partículas × N bits, pbest, gbest          │       │
│  └──────────────────┬───────────────────────────┘       │
│                     │                                   │
│           ┌────────▼────────┐                           │
│           │  t = 1, ..., T  │ ◄───────────────┐        │
│           └────────┬────────┘                 │        │
│                    │                           │        │
│  ┌─────────────────▼─────────────────────┐    │        │
│  │  Passo 1: Avaliar Fitness (SHAP)      │    │        │
│  │  F(b) = w_p·P̂(b) + w_e·Ê(b)          │    │        │
│  │  Atualizar pbest, gbest               │    │        │
│  └─────────────────┬─────────────────────┘    │        │
│                    │                           │        │
│  ┌─────────────────▼─────────────────────┐    │        │
│  │  Passo 2: Calcular mbest, α(t)        │    │        │
│  │  mbest = média dos pbest              │    │        │
│  │  α = 1.0 - 0.5 * (t/T)               │    │        │
│  └─────────────────┬─────────────────────┘    │        │
│                    │                           │        │
│  ┌─────────────────▼─────────────────────┐    │        │
│  │  Passo 3: Atualização QUÂNTICA        │    │        │
│  │  Schrödinger → Sigmoid → Colapso      │    │        │
│  │  x = p ± α|mbest - x|·ln(1/u)        │    │        │
│  │  b = Bernoulli(σ(x))                  │    │        │
│  └─────────────────┬─────────────────────┘    │        │
│                    │                           │        │
│  ┌─────────────────▼─────────────────────┐    │        │
│  │  Passo 4: Mutação GENÉTICA (GA)       │    │        │
│  │  Flip bits nas piores partículas      │    │        │
│  └─────────────────┬─────────────────────┘    │        │
│                    │                           │        │
│  ┌─────────────────▼─────────────────────┐    │        │
│  │  Passo 5: Reparação de Restrições     │    │        │
│  │  Garantir Σb_i ≥ N_min                │    │        │
│  └─────────────────┬─────────────────────┘    │        │
│                    │                           │        │
│           ┌────────▼────────┐                 │        │
│           │  t < T ?        ├─── Sim ─────────┘        │
│           └────────┬────────┘                          │
│                    │ Não                                │
│  ┌─────────────────▼─────────────────────┐              │
│  │  Retornar gbest, gbest_fit, histórico │              │
│  └───────────────────────────────────────┘              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Versão Enhanced: Multi-Estratégia e Operadores Avançados

A versão **Enhanced** do QPSO Híbrido adiciona cinco mecanismos sofisticados para melhorar a qualidade da solução:

### 9.1 Multi-Estratégia de Atualização

Cada partícula aplica **uma de duas estratégias** em cada iteração:

**Estratégia A — QPSO Global (60% de probabilidade):**

A estratégia padrão do QPSO (seção 5.9), com poço delta centrado entre pbest e gbest:

$$
p_{i,j} = \varphi \cdot \text{pbest}_{i,j} + (1-\varphi) \cdot \text{gbest}_j
$$

$$
L_{i,j} = \alpha \cdot |\overline{p}_j - b_{i,j}|
$$

**Estratégia B — Busca Local em pbest (40% de probabilidade):**

O atrator é deslocado para orbitar o **melhor pessoal**, reduzindo convergência global e focando em exploração local:

$$
p_{i,j} = \varphi' \cdot \text{pbest}_{i,j} + (1-\varphi') \cdot b_{i,j}^{(t)}, \quad \varphi' \sim U(0.5, 1)
$$

$$
L_{i,j} = \alpha \cdot 0.5 \cdot |\text{pbest}_{i,j} - b_{i,j}^{(t)}|
$$

Nota: $L$ é reduzido pela metade, resultando em perturbações menores (refinamento fino).

### 9.2 Lévy Flight Mutation (Mantegna, 1994)

O **voo de Lévy** gera passos com **cauda pesada**, permitindo saltos longos ocasionais que exploram regiões distantes do espaço de busca.

**Distribuição de Lévy:**

$$
\text{Lévy}(\beta) \sim \frac{u}{|v|^{1/\beta}}
$$

onde $u \sim \mathcal{N}(0, \sigma_u^2)$, $v \sim \mathcal{N}(0, 1)$ e:

$$
\sigma_u = \left[\frac{\Gamma(1+\beta) \cdot \sin\left(\frac{\pi\beta}{2}\right)}{\Gamma\left(\frac{1+\beta}{2}\right) \cdot \beta \cdot 2^{(\beta-1)/2}}\right]^{1/\beta}
$$

com $\beta = 1.5$ (índice de estabilidade).

**Aplicação:** O número de bits a inverter é determinado pelo passo de Lévy:

$$
n_{\text{flips}} = \min\left(\max(1, \lfloor |\text{Lévy}(1.5)| \rfloor), N - N_{\min}\right)
$$

A mutação Lévy é aplicada à **fração inferior** (pior fitness) da população com taxa decrescente. **Nota de implementação:** a fração varia entre as versões — o QPSO discreto legado (`qpso_ml_standalone.py`) usa a **metade** inferior (`num_particles // 2`); o QPSO contínuo legado (`qpso_layout_standalone.py`) usa **um terço** (`num_particles // 3`); o núcleo refatorado (`qpso_continuo.py`) também usa um terço. A taxa de aplicação é:

$$
p_{\text{Lévy}}(t) = \mu_{\text{Lévy}} \cdot \left(1 - 0.5 \cdot \frac{t}{T}\right)
$$

### 9.3 Crossover Uniforme com gbest

Um operador de **crossover** combina bits de uma partícula aleatória com o gbest:

$$
\text{child}_j = \begin{cases}
\text{gbest}_j & \text{se } m_j = 1 \\
b_{k,j}^{(t)} & \text{se } m_j = 0
\end{cases}, \quad m_j \sim \text{Bernoulli}(0.5)
$$

onde $k$ é uma partícula-pai selecionada aleatoriamente.

O filho substitui a **pior** partícula somente se seu fitness é superior:

$$
b_{\text{worst}}^{(t)} \leftarrow \text{child} \quad \text{se} \quad F(\text{child}) > F(b_{\text{worst}}^{(t)})
$$

A taxa de crossover decai:

$$
p_{cx}(t) = p_{cx}^{(0)} \cdot \left(1 - \frac{t}{T} + 0.1\right)
$$

### 9.4 Elitismo

As **top-$K$** partículas (melhores fitness) são preservadas intactas em cada iteração, impedindo que operadores de mutação ou Lévy as destruam:

$$
K = \min\left(3, M\right)
$$

As $K$ melhores partículas são copiadas e reinseridas na população após todos os operadores.

### 9.5 Reparação Inteligente

Em vez de ativar bits aleatórios para satisfazer $\sum b_i \geq N_{\min}$, a reparação **prioriza boias com maior frequência entre os pbest** (boias que o enxame "concorda" que são boas):

$$
\text{score}_j = \frac{1}{M} \sum_{i=1}^{M} \text{pbest}_{i,j}
$$

Os bits inativos com maior score são ativados primeiro.

### 9.6 Resumo dos Operadores Enhanced

| Operador | Probabilidade/Frequência | Objetivo |
|:---|:---|:---|
| QPSO Global | 60% por partícula | Exploração ampla (Schrödinger) |
| Busca Local | 40% por partícula | Refinamento fino em pbest |
| Lévy Flight | ~5% na metade ruim | Saltos longos exploratórios |
| Crossover | ~30% decrescente | Transferência de blocos do gbest |
| Mutação GA | Adaptativa (35%→5%) | Escape de ótimos locais |
| Elitismo | Top-3 preservados | Protege melhores soluções |
| Reparação Inteligente | 100% (garantia) | Viabilidade da restrição |

---

## 10. Função de Fitness Multi-Objetivo

### 10.1 Formulação com SHAP

$$
F(\mathbf{b}) = w_p \cdot \hat{P}(\mathbf{b}) + w_e \cdot E(\mathbf{b})
$$

onde:

$$
\hat{P}(\mathbf{b}) = \phi_0 + \sum_{i=1}^{N} b_i \cdot c_i
$$

$$
E(\mathbf{b}) = \frac{\hat{P}(\mathbf{b})}{\sum_{i=1}^{N} b_i}
$$

**Expandindo:**

$$
F(\mathbf{b}) = w_p \left(\phi_0 + \sum_{i=1}^{N} b_i c_i\right) + w_e \cdot \frac{\phi_0 + \sum_{i=1}^{N} b_i c_i}{\sum_{i=1}^{N} b_i}
$$

$$
F(\mathbf{b}) = \left(w_p + \frac{w_e}{n_a}\right) \cdot \left(\phi_0 + \sum_{i \in \mathcal{A}} c_i\right)
$$

onde $n_a = |\mathcal{A}|$ é o número de boias ativas.

### 10.2 Penalização de Inviabilidade

Soluções que violam a restrição de mínimo recebem penalidade extrema:

$$
F(\mathbf{b}) = -10^6 \quad \text{se} \quad \sum_{i=1}^{N} b_i < N_{\min}
$$

### 10.3 Interpretação dos Pesos

| Configuração | $w_p$ | $w_e$ | Comportamento |
|:---|:---|:---|:---|
| Foco em potência | 0.95 | 0.05 | Maximiza potência bruta |
| **Balanceado** | **0.70** | **0.30** | **Equilíbrio potência/eficiência** |
| Foco em eficiência | 0.30 | 0.70 | Prioriza redução de boias |

---

## 11. Análise de Convergência

### 11.1 Convergência Teórica do QPSO

Sun, Feng & Xu (2004) demonstraram que o QPSO satisfaz a condição de convergência global sob certas condições:

**Teorema (Sun et al., 2004):** Se o coeficiente $\alpha^{(t)}$ satisfaz:

$$
\lim_{t \to \infty} \alpha^{(t)} = 0, \quad \sum_{t=0}^{\infty} \alpha^{(t)} = \infty
$$

então a sequência $\{\mathbf{x}_i^{(t)}\}$ converge para o ótimo global com probabilidade 1.

**Nota sobre a implementação:** No caso linear $\alpha^{(t)} = 1.0 - 0.5 \cdot (t/T)$ com horizonte finito $T=80$, temos $\alpha_{\min} = 0.5 > 0$, o que **não satisfaz formalmente** a condição $\alpha \to 0$. Portanto, a convergência **ao ótimo global** não é formalmente garantida. No entanto, o decaimento de $\alpha$ de 1.0 a 0.5 **favorece** a transição progressiva de exploração para exploitação, e a combinação com operadores de Lévy, mutação GA e crossover compensa a ausência da garantia formal, resultando em boa convergência empírica.

**Distinção importante (duas noções de convergência — não confundir).**
1. **Convergência do enxame** (a posição de cada partícula converge, i.e. o raio de busca $L \to 0$): o critério padrão de estabilidade do QPSO (Sun, Xu & Feng) é $\beta \equiv \alpha < e^{\gamma} \approx 1.781$, onde $\gamma \approx 0.5772$ é a constante de Euler–Mascheroni. Como a implementação usa $\alpha \in [0.5,\,1.0]$, **esse critério é satisfeito** — o enxame contrai e estabiliza.
2. **Convergência ao ótimo global com probabilidade 1**: exige exploração assintótica infinita, daí a condição $\alpha \to 0$ com $\sum_t \alpha^{(t)} = \infty$. Com $\alpha_{\min} = 0.5 > 0$ e horizonte finito, **essa garantia mais forte não vale** formalmente.

Ou seja, a contração do enxame está assegurada pelo critério (1); o que não se garante é atingir o ótimo global (2). Recomenda-se conferir o enunciado exato dos teoremas nas referências [1] e [2] antes de citá-los na dissertação.

### 11.2 Complexidade Computacional

| Componente | Complexidade por iteração |
|:---|:---|
| Avaliação Fitness (SHAP) | $\mathcal{O}(M \cdot N)$ |
| Atualização Quântica | $\mathcal{O}(M \cdot N)$ |
| Mutação GA | $\mathcal{O}(M \cdot N)$ |
| Lévy Flight (Enhanced) | $\mathcal{O}(M \cdot N)$ |
| Reparação | $\mathcal{O}(M \cdot N)$ |
| **Total por iteração** | **$\mathcal{O}(M \cdot N)$** |
| **Total do algoritmo** | **$\mathcal{O}(T \cdot M \cdot N)$** |

Para $T = 80$, $M = 30$, $N = 49$: são $T \cdot M = 2.400$ **avaliações de fitness** (a implementação legada avalia 2× por iteração — pós-movimento e pós-mutação —, totalizando ~4.800). O produto $T \cdot M \cdot N = 117.600$ corresponde ao número de **operações elementares** (por bit), refletindo o custo total $\mathcal{O}(T \cdot M \cdot N)$ — não ao número de avaliações de fitness.

### 11.3 Balanço Exploração × Exploitação

O QPSO Híbrido equilibra exploração e exploitação (intensificação) em múltiplos níveis:

```
     EXPLORAÇÃO                      EXPLOITAÇÃO
     (Diversidade)                   (Convergência)
         │                                │
    ┌────▼────┐                      ┌────▼────┐
    │  α alto │ ─── Início ───────── │  α baixo│
    └─────────┘                      └─────────┘
    ┌─────────┐                      ┌─────────┐
    │Lévy     │ ─── Saltos longos ── │Busca    │
    │Flight   │                      │Local    │
    └─────────┘                      └─────────┘
    ┌─────────┐                      ┌─────────┐
    │Mutação  │ ─── p_mut alto ───── │p_mut    │
    │Genética │     (início)         │baixo    │
    └─────────┘                      └─────────┘
    ┌─────────┐                      ┌─────────┐
    │Crossover│ ─── Troca blocos ──  │Elitismo │
    │c/ gbest │                      │Top-K    │
    └─────────┘                      └─────────┘
```

---

## 12. Comparativo com Outros Algoritmos Híbridos (QAOA, QEA)

### 12.1 QAOA Clássico-Simulado Híbrido

O **Quantum Approximate Optimization Algorithm** (Farhi et al., 2014) é um algoritmo quântico variacional para problemas combinatórios. Na versão clássica simulada:

**Hamiltoniano de Problema (Fase de Busca):**

$$
U_C(\gamma) = e^{-i\gamma H_C}
$$

Simulado classicamente via busca local gulosa com probabilidade $\gamma$: testar inversão de cada bit e aceitar se melhorar o fitness.

**Hamiltoniano de Mixing (Fase de Perturbação):**

$$
U_B(\beta) = e^{-i\beta H_B}
$$

Simulado via flips aleatórios com probabilidade $\beta \cdot 0.15$ em cada bit.

**Schedule:** $\gamma$ decai linearmente ($0.9 \to 0.3$) e $\beta$ cresce ($0.1 \to 0.5$) ao longo de $p$ camadas.

### 12.2 QEA Híbrido (Quantum Evolutionary Algorithm)

O **QEA** (Han & Kim, 2002) utiliza **Q-bits** como representação probabilística:

**Cromossomo Quântico:**

$$
\mathbf{q}_i = \begin{bmatrix} \alpha_{i,1} & \alpha_{i,2} & \cdots & \alpha_{i,N} \\ \beta_{i,1} & \beta_{i,2} & \cdots & \beta_{i,N} \end{bmatrix}
$$

com $|\alpha_{i,j}|^2 + |\beta_{i,j}|^2 = 1$ (normalização de Born).

**Colapso:** $b_{i,j} = 1$ se $r < |\beta_{i,j}|^2$, caso contrário $b_{i,j} = 0$.

**Rotação Quântica (Porta de Rotação):**

$$
\begin{bmatrix} \alpha'_{i,j} \\ \beta'_{i,j} \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} \alpha_{i,j} \\ \beta_{i,j} \end{bmatrix}
$$

onde $\theta$ é determinado pela **tabela de lookup:**

| $x_{i,j}$ | $\text{gbest}_j$ | $\theta$ |
|:---:|:---:|:---:|
| 0 | 1 | $+\theta_{\text{base}}$ |
| 1 | 0 | $-\theta_{\text{base}}$ |
| 0 | 0 | 0 |
| 1 | 1 | 0 |

com $\theta_{\text{base}} = 0.05\pi$ (intensidade média).

### 12.3 Comparativo Teórico

| Característica | QAOA Híbrido | QEA Híbrido | **QPSO Híbrido** |
|:---|:---|:---|:---|
| Inspiração Quântica | Circuito Variacional | Q-bits (superposição) | **Poço Delta (Schrödinger)** |
| Representação | Binária direta | Quântica (α, β) | **Binária via Sigmoid** |
| Exploração | Schedule γ/β | Rotação θ | **Tunneling + Lévy** |
| Exploitação | Busca local gulosa | Injeção de elite | **Atrator pbest/gbest** |
| Hibridização | Mutação GA | Mutação GA + Perturbação | **Mutação + Crossover + Lévy** |
| Memória | gbest apenas | gbest + Q-pop | **pbest + gbest + mbest** |
| Parâmetros | $\gamma, \beta, p$ | $\theta, \epsilon$ | **$\alpha, p_{\text{mut}}, \mu_{\text{Lévy}}$** |
| Convergência | Por camadas ($p$) | Por iterações ($T$) | **Por iterações ($T$)** |

### 12.4 Justificativa do QPSO como Melhor Candidato

1. **Sem velocidade:** O QPSO elimina o parâmetro de velocidade do PSO, reduzindo complexidade de calibração.
2. **Distribuição Laplace:** O tunneling quântico gera distribuição de cauda pesada, superior à distribuição normal para escape de ótimos locais.
3. **Mbest como centroide:** A média das melhores posições funciona como um guia coletivo do enxame, incorporando informação de todas as partículas.
4. **Flexibilidade de discretização:** A sigmoid preserva gradiente suave, ao contrário do colapso discreto abrupto do QEA.

---

## 13. Referências

1. **Sun, J., Feng, B., & Xu, W.** (2004). *Particle Swarm Optimization with Particles Having Quantum Behavior*. IEEE Congress on Evolutionary Computation. — Artigo fundacional do QPSO.

2. **Sun, J., Xu, W., & Feng, B.** (2004). *A Global Search Strategy of Quantum-Behaved Particle Swarm Optimization*. IEEE Conference on Cybernetics and Intelligent Systems. — Prova de convergência global.

3. **Kennedy, J., & Eberhart, R.** (1995). *Particle Swarm Optimization*. IEEE International Conference on Neural Networks. — PSO clássico.

4. **Han, K.H., & Kim, J.H.** (2002). *Quantum-Inspired Evolutionary Algorithm for a Class of Combinatorial Optimization*. IEEE Trans. Evolutionary Computation. — QEA com rotação de Q-bits.

5. **Farhi, E., Goldstone, J., & Gutmann, S.** (2014). *A Quantum Approximate Optimization Algorithm*. arXiv:1411.4028. — QAOA original.

6. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5-32. — Random Forest Regressor.

7. **Lundberg, S.M., & Lee, S.I.** (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. — Framework SHAP.

8. **Lundberg, S.M. et al.** (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence. — TreeExplainer (SHAP exato para árvores).

9. **Mantegna, R.N.** (1994). *Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes*. Physical Review E. — Algoritmo para Lévy flights.

10. **Shapley, L.S.** (1953). *A Value for N-Person Games*. Contributions to the Theory of Games. — Teoria de valores de Shapley.

11. **Holland, J.H.** (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press. — Algoritmos Genéticos (mutação, crossover).

12. **Clerc, M., & Kennedy, J.** (2002). *The particle swarm—explosion, stability, and convergence in a multidimensional complex space*. IEEE Trans. Evolutionary Computation. — Análise de convergência do PSO.

---

## Glossário de Notações

| Símbolo | Significado |
|:---|:---|
| $N$ | Número total de WECs (49) |
| $M$ | Número de partículas no enxame |
| $T$ | Número máximo de iterações |
| $\mathbf{b}_i^{(t)}$ | Vetor binário da partícula $i$ na iteração $t$ |
| $b_{i,j}$ | Bit $j$ da partícula $i$ (0 = desligada, 1 = ativa) |
| $\text{pbest}_i$ | Melhor posição pessoal da partícula $i$ |
| $\text{gbest}$ | Melhor posição global do enxame |
| $\overline{\mathbf{p}}$ | Mean Best Position (mbest) |
| $\alpha^{(t)}$ | Coeficiente de contração-expansão |
| $L_{i,j}$ | Comprimento característico do poço de potencial |
| $\sigma(\cdot)$ | Função sigmoid (logística) |
| $\phi_0$ | Valor base SHAP ($\mathbb{E}[\hat{f}]$) |
| $c_i$ | Contribuição SHAP da boia $i$ |
| $\hat{P}(\mathbf{b})$ | Potência predita via decomposição SHAP |
| $F(\mathbf{b})$ | Função de fitness multi-objetivo |
| $p_{\text{mut}}(t)$ | Taxa de mutação adaptativa na iteração $t$ |
| $\Gamma(\cdot)$ | Função Gamma de Euler |
| $\mathcal{N}(\mu, \sigma^2)$ | Distribuição Normal |
| $U(a, b)$ | Distribuição Uniforme contínua |
| $N_{\min}$ | Restrição de mínimo de boias ativas |

---

*Documento gerado para o projeto de Mestrado — UFPR, 2026.*
*Última atualização: Abril 2026.*
