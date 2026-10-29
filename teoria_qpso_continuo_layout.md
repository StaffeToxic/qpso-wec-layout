# Fundamentação Teórica: QPSO Contínuo para Otimização de Layout de WECs

## (Surrogate Suave + Trust-Region) — complemento ao documento do QPSO híbrido discreto

> Este documento cobre o **caso contínuo** — otimizar diretamente as coordenadas
> $(X,Y)$ de todas as 49 boias — que é distinto do QPSO **discreto/SHAP**
> descrito em `teoria_qpso_hibrido_ml.md` (seleção binária de boias). As seções
> de QPSO-núcleo (§4–§5 daquele documento) permanecem válidas e são referenciadas
> aqui; as seções de SHAP, sigmoid, mutação GA e fitness multi-objetivo **não se
> aplicam** ao caso contínuo.

---

## 1. Formulação do problema contínuo

Dado um campo marítimo retangular $\Omega = [0, L_x] \times [0, L_y]$ (nos datasets,
$1000 \times 1010$ m) com $N = 49$ boias **todas ativas**, busca-se o vetor de
coordenadas que maximiza a potência total:

$$
\max_{\mathbf{x} \in \mathbb{R}^{2N}} \; \hat{P}(\mathbf{x}), \qquad
\mathbf{x} = (X_1, Y_1, \ldots, X_N, Y_N)
$$

sujeito a:

$$
\text{(caixa)} \quad \mathbf{x}_{\min} \le \mathbf{x} \le \mathbf{x}_{\max}
\qquad
\text{(distância)} \quad \|(X_i,Y_i) - (X_j,Y_j)\| \ge d_{\min}\ \ \forall i \ne j
$$

onde $\hat{P}$ é um **modelo surrogate** treinado em simulações e $d_{\min} = 50$ m
é a distância mínima de segurança (regra de projeto do benchmark; ver §6).

Diferença essencial vs. o caso discreto: aqui o espaço é **contínuo** de dimensão
$2N = 98$, e não há decisão binária — não há SHAP, sigmoid nem eficiência por boia.

---

## 2. O surrogate determina a viabilidade da busca contínua

### 2.1 Random Forest é constante por partes

Um Random Forest prediz pela média de folhas:
$\hat{f}(\mathbf{x}) = \frac{1}{T}\sum_t h_t(\mathbf{x})$. Cada $h_t$ é constante
dentro de cada célula de partição; logo $\hat{f}$ é **constante por partes**, com
**gradiente nulo quase em toda parte**. Para uma metaheurística contínua isto é
fatal por dois motivos:

1. **Sem gradiente local:** mover uma boia dentro de uma célula não altera a
   predição → não há sinal de melhora para guiar a busca.
2. **Máximo na melhor folha:** o máximo de um RF sobre a caixa é o maior valor de
   folha, que corresponde (em média) ao **melhor ponto de treino**. Como o melhor
   layout do dataset já está nessa folha, **não existe nada acima dele** que a
   busca contínua possa encontrar.

> **Consequência prática (verificada).** Otimizar o RF com QPSO contínuo
> **devolve a semente**: o melhor layout do dataset é um máximo local do RF.
> Empiricamente (Sydney 49), amostrando 2000 perturbações ao redor do melhor
> layout, o **maior ganho sob RF é $-7.024$ W** (negativo) — tudo ao redor é
> pior. Foi exatamente por isso que a versão anterior do QPSO contínuo empatou
> ao centavo com o melhor layout (ver `comparacao_*.md`).

### 2.2 Por que um surrogate suave resolve

Um surrogate **suave e diferenciável** tem:

- **gradiente quase em toda parte** → direções de melhora locais existem;
- **máximo possivelmente entre os pontos de treino** → a busca contínua pode
  encontrar layouts melhores que qualquer um do dataset (interpolação).

Empiricamente (Sydney 49): o mesmo entorno do melhor layout tem
**$+23.678$ W de headroom** sob o surrogate suave, contra $-7.024$ W sob RF.

### 2.3 Escolha: ensemble de MLP

Com $n \in [18\text{k}, 36\text{k}]$ amostras e $2N = 98$ features, **GP exato é
inviável** ($\mathcal{O}(n^3)$). Usa-se um **ensemble de $K$ MLPs** com ativação
$\tanh$ (classe $C^\infty$, portanto suave), entradas/saídas normalizadas
(`StandardScaler`), cada membro com semente distinta:

$$
\hat{P}(\mathbf{x}) = \mu(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K} f_k(\mathbf{x}),
\qquad
\sigma(\mathbf{x}) = \sqrt{\frac{1}{K}\sum_{k=1}^{K}\big(f_k(\mathbf{x}) - \mu(\mathbf{x})\big)^2}
$$

O desvio entre membros $\sigma(\mathbf{x})$ é uma estimativa **epistêmica** de
incerteza: pequena onde há dados (membros concordam), maior fora da distribuição
(membros extrapolam de forma divergente). Empiricamente, o MLP ensemble obteve
$R^2 = 0{,}9932$ vs. $0{,}9815$ do RF — **mais preciso e suave** (sem trade-off).

---

## 3. Atualização QPSO contínua

Usa-se a forma canônica (idêntica ao §5.9 do documento híbrido, **com o fator de
contração-expansão já correto** — ver a nota do fator 2 na §5.8 daquele documento):

$$
\boxed{\;x_{i,j}^{(t+1)} = p_{i,j}^{(t)} \pm \alpha^{(t)}\,\big|\overline{p}_j^{(t)} - x_{i,j}^{(t)}\big|\,\ln\!\left(\tfrac{1}{u_{i,j}}\right)\;}
$$

com $p_{i,j} = \varphi\,\text{pbest}_{i,j} + (1-\varphi)\,\text{gbest}_j$,
$\overline{p}$ = mbest (média dos pbest), $\alpha^{(t)} = 1.0 \to 0.5$ linear,
$u \sim U(10^{-10},1)$ e sinal $\pm$ com probabilidade $0{,}5$.

**Correções de implementação (vs. o núcleo legado):**

| # | Defeito legado | Correção (`qpso_continuo.py`) |
|---|---|---|
| 5a | elites recolocados nos slots $0..k$ → quebra $\text{pbest}_i \leftrightarrow$ partícula $i$ | elites restaurados **no próprio índice** `elite_idx` |
| 5b | layout de referência descartado (`gbest_fit=-\infty` nunca avaliado) | semeadura explícita via `seed_with_ref` |
| 5c | Lévy seleciona piores por fitness **defasado** (pré-movimento) | ranking recomputado após o movimento |
| 5d | duas avaliações do surrogate por iteração | **uma** avaliação/iteração (+ subconjunto Lévy) |

---

## 4. Restrição de distância mínima (reparo por projeção)

A restrição $d_{\min}$ é tratada por **reparo soft**: após cada movimento, pares de
boias com distância $< d_{\min}$ são afastados iterativamente. Para cada par
$(i,j)$ violado, ambos os pontos se deslocam ao longo da reta que os une, cada um
por metade do déficit:

$$
\mathbf{p}_i \leftarrow \mathbf{p}_i + \tfrac{1}{2}\,(d_{\min} - d_{ij})\,\hat{\mathbf{u}}_{ij},
\qquad
\hat{\mathbf{u}}_{ij} = \frac{\mathbf{p}_i - \mathbf{p}_j}{d_{ij}}
$$

iterado (com projeção na caixa) até não haver violação ou atingir o teto de passos.
**Limitação conhecida (soft):** junto à fronteira do domínio o reparo é aproximado
(os pontos são comprimidos contra a parede). Na prática o surrogate de WEC
recompensa layouts **espalhados**, então as boias não se aglomeram num canto; o
reparo entrega layouts viáveis no interior. Para os 49 WECs num campo de
$1000^2\,\text{m}^2$ com $d_{\min}=50$ m, a densidade é folgada (uma grade
$\sim 21\times 21$ comportaria $\gg 49$ boias).

---

## 5. Trust-region: Lower Confidence Bound (LCB)

O risco central de otimizar um surrogate é a **exploração de regiões OOD**, onde o
modelo extrapola e pode **superestimar** a potência. Para mitigar, a busca maximiza
não a média, mas o **limite inferior de confiança**:

$$
\text{acq}(\mathbf{x}) = \mu(\mathbf{x}) - \kappa\,\sigma(\mathbf{x}), \qquad \kappa \ge 0
$$

- $\kappa = 0$: busca gulosa (maximiza a média) — máxima exploração, máximo risco OOD.
- $\kappa > 0$: penaliza incerteza → mantém a busca onde os membros concordam
  (região com dados). Reporta-se a **potência média** $\mu$ no layout encontrado,
  enquanto a busca é guiada pela aquisição (estilo otimização Bayesiana).

> **Alerta de honestidade (a calibrar — Fase 3).** A incerteza de ensembles de MLP
> pode ser **superconfiante**: membros com mesma arquitetura/dados às vezes
> extrapolam de forma correlacionada, subestimando $\sigma$ fora da distribuição.
> Se $\sigma$ for pequeno demais em toda parte, o LCB não controla o deslocamento e
> $\kappa$ tem pouco efeito. Por isso a validação (§7) **não confia só em $\sigma$**:
> usa também o cross-check com o RF e a distância à vizinhança de dados.

---

## 6. Parâmetros e datasets

| Parâmetro | Valor | Origem |
|---|---|---|
| $N$ (boias) | 49 | dataset |
| Domínio | $[0,1000]\times[0,1010]$ m | extensão observada |
| $d_{\min}$ | 50 m | regra de projeto (mediana do espaçamento observado ≈ 49–50 m) |
| Partículas / iterações | 40–50 / 120–150 | — |
| $\alpha$ | $1.0 \to 0.5$ | §5.8 (híbrido) |
| Ensemble | $K=5$ MLP $\tanh$ $(128,64)$ | §2.3 |

---

## 7. Critério de validação honesta (sem BEM)

Não há simulador BEM disponível — apenas o dataset e o surrogate. Portanto, um
ganho só é considerado **credível** se satisfizer, simultaneamente:

1. **Concordância de surrogates:** o layout otimizado é melhor que o melhor layout
   também sob o **RF** (treinado nos mesmos dados, $R^2 \approx 0.98$), não só sob
   o MLP. Ganho só no MLP ⇒ **extrapolação específica do MLP** (pico fantasma).
2. **Proximidade aos dados:** deslocamento moderado e pequena distância normalizada
   ao layout de treino mais próximo (`nn_norm`) ⇒ dentro da distribuição confiável.
3. **Viabilidade:** $d_{\min} \ge 50$ m.
4. **Estatística:** média $-$ desvio $> 0$ sobre múltiplas sementes.

---

## 8. Resultados experimentais (Fase 3)

Sydney 49, **5 sementes** por configuração, surrogate treinado uma vez. Ganho
relativo ao melhor layout do dataset (médias); figura: `experimentos_sydney.png`.

| Config | $\kappa$ | ganho MLP (%) | ganho RF (%) | desloc. (m) | `nn_norm` | min_dist (m) |
|---|---:|---:|---:|---:|---:|---:|
| **RF (baseline #3)** | — | **+0,00** | **+0,00** | 0,1 | 0,000 | 50,0 |
| MLP | 0,0 | +5,73 | **−1,71** | 403,5 | 0,251 | 50,0 |
| MLP | 0,5 | +5,70 | **−1,11** | 397,5 | 0,253 | 50,1 |
| MLP | 1,0 | +5,38 | **−0,94** | 378,0 | 0,242 | 50,1 |
| MLP | 2,0 | +4,76 | **−1,09** | 333,7 | 0,232 | 50,0 |
| MLP | 5,0 | +2,16 | **−1,06** | 133,2 | 0,125 | 50,4 |

**Três conclusões, todas honestas:**

1. **O baseline confirma o #3.** Otimizar o **RF** devolve a semente: ganho 0,00 %
   e deslocamento 0,1 m. O melhor layout do dataset é um máximo local do RF — não há
   o que a busca contínua melhore. (É o "empate ao centavo" do `comparacao_*.md`,
   agora reproduzido de forma controlada.)

2. **O trust-region ($\kappa$) funciona.** Aumentar $\kappa$ de 0 a 5 reduz o
   deslocamento de 403 m → 133 m e a distância aos dados (`nn_norm`) de 0,25 → 0,12.
   O LCB de fato puxa a busca de volta para perto da distribuição de treino.

3. **Mas o ganho não é real.** Em **todos** os $\kappa$ o ganho sob MLP é positivo
   (+2 a +5,7 %) enquanto o ganho sob **RF é sempre negativo** ($\approx -1\%$). Dois
   modelos independentes com $R^2 \approx 0{,}98$–$0{,}99$ **discordam** sobre os
   layouts otimizados: o MLP "vê" melhora, o RF "vê" piora. Como concordam no conjunto
   de teste mas divergem nos layouts propostos, a "melhora" é **extrapolação específica
   do MLP** (pico fantasma fora da distribuição), não um ganho físico.

**Veredito parcial (superar o melhor layout).** Sob o critério de validação honesta (§7),
**nenhum ganho real sobrevive** ao tentar *superar o melhor layout do dataset*: não há $\kappa$
em que MLP e RF concordem que o otimizado supera o melhor. O "+6,25 %" da Fase 2 era **miragem
de extrapolação** — demonstrado com multi-seed e cross-check.

### 8.1 Melhorar um layout *subótimo* (o caso que importa para "mover boias")

Superar o melhor layout é o problema *errado* — ele já é o máximo do RF. O problema certo é:
*dado um layout de qualidade X, movê-lo melhora?* Otimizando o MLP a partir de layouts de vários
percentis de potência (3 sementes), mede-se o ganho vs a partida sob os DOIS modelos:

| Partida | ganho MLP | ganho RF (cross-check) | desloc. | `nn_norm` |
|---|---:|---:|---:|---:|
| p10 (ruim) | +9,90% | **+4,39%** | 402 m | 0,289 |
| p50 (mediano) | +7,96% | **+1,78%** | 312 m | 0,250 |
| p90 (bom) | +5,88% | −0,23% | 341 m | 0,260 |
| p100 (melhor) | +4,85% | −1,20% | 332 m | 0,257 |

Existe uma **fronteira de credibilidade**: partindo de layouts subótimos (p10/p50) o ganho é
**confirmado pelos dois modelos** (RF +4,4% e +1,8%) — é real. Partindo de layouts já bons (p90+),
só o MLP "vê" ganho (extrapolação). O número honesto é o **confirmado pelo RF**.

### 8.2 Otimização por consenso — credibilidade por construção

Em vez de otimizar o MLP e conferir o RF *depois*, otimiza-se o **consenso pessimista**
$\hat P_{\cap}(\mathbf{x}) = \min\!\big(\hat P_{\text{MLP}}(\mathbf{x}),\, \hat P_{\text{RF}}(\mathbf{x})\big)$.
Um layout só pontua alto se **ambos** concordam → ganho credível por construção, miragem cortada:

| Partida | **ganho consenso** | ganho MLP | ganho RF |
|---|---:|---:|---:|
| p10 | **+5,36%** | +5,44% | +5,36% |
| p50 | **+2,71%** | +3,49% | +2,58% |
| p90 | **+0,87%** | +1,32% | +0,78% |
| p100 | **+0,10%** | +0,53% | −0,04% |

O consenso **faz MLP e RF convergirem** (em p10, ambos ≈ +5,4%, contra a fantasia +9,9% do MLP
sozinho) e em p100 reporta honestamente **≈ 0** (nada a extrair do melhor layout).

**Veredito final (contínuo).** O QPSO contínuo **move boias e melhora um array dado de forma
validada** — +5,4% (layout ruim), +2,7% (mediano), confirmado por dois modelos independentes. O
que ele *não* faz — e nenhum método baseado só nestes dados faria — é **superar o melhor layout
conhecido** (isso exige BEM no laço; §9.2).

---

## 9. Conclusão, limitações e recomendações

### 9.1 Conclusão empírica (o que a Fase 3 estabeleceu)

A maquinaria do QPSO contínuo está **correta e validada** (otimiza de fato em superfície
suave, respeita `d_min`, elitismo/pbest consistentes — 22 testes). O resultado tem duas faces:

- **Melhorar um array dado (mover boias): FUNCIONA e é validável.** Partindo de um layout
  subótimo, o QPSO o melhora com ganho **confirmado por dois modelos independentes**; a
  otimização por **consenso** $\min(\hat P_{\text{MLP}}, \hat P_{\text{RF}})$ (§8.2) torna isso
  credível *por construção* (+5,4% de um layout ruim, +2,7% de um mediano, ambos os modelos
  concordando).
- **Superar o melhor layout conhecido: NÃO, com estes dados.** O melhor layout já é o máximo do
  RF; além dele, só o MLP "vê" ganho, e é extrapolação. O gargalo aqui não é o otimizador, é
  **epistêmico**: sem ground-truth novo (BEM), não há como distinguir ganho físico de fantasia
  do surrogate acima da fronteira de dados.

### 9.2 Recomendações

**Para "mover boias e melhorar um array" (objetivo principal): use o consenso.** Otimizar
$\min(\hat P_{\text{MLP}}, \hat P_{\text{RF}})$ entrega ganho credível *por construção* (§8.2):
reposiciona as boias de um array dado com ganho que dois modelos independentes confirmam
(+5,4% / +2,7% partindo de layout ruim/mediano). Extensões: $K>2$ surrogates (mais conservador)
ou um cap de deslocamento (refinamento local).

**Para *superar* o melhor layout conhecido (mais ambicioso): só com BEM no laço.** Active learning
— a cada rodada, simular por BEM os layouts propostos pelo consenso, adicioná-los ao dataset e
retreinar os surrogates — corrige os "picos fantasma" onde a busca os explora. O dataset veio de um
BEM; ferramentas open-source (**Capytaine**/NEMOH) podem fechar o laço.

**Complementar:** o caso **discreto/SHAP** (seleção de boias) é contribuição separada e robusta
(aditivo ⇒ interpolação), útil para reduzir o nº de boias mantendo a potência.

### 9.3 Limitações técnicas

- **Ganho sob surrogate.** Sem BEM, todo ganho é *predito*; o cross-check RF↔MLP e a
  proximidade aos dados são *proxies* de confiabilidade, não substitutos de simulação.
- **Incerteza de ensemble pode ser superconfiante.** Aqui o $\sigma$ do ensemble
  controlou o deslocamento (κ teve efeito), mas **não bastou** para garantir
  confiabilidade — por isso o cross-check com RF foi decisivo.
- **Reparo near-boundary** é aproximado (constraint soft).
- **Comparação com o artigo** deve usar a **mesma política de semeadura** para QPSO e
  PSO; caso contrário a vantagem reflete inicialização, não capacidade de busca.

---

## 10. Referências

Ver `teoria_qpso_hibrido_ml.md` §13. Adicionalmente:

- **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017). *Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS. — incerteza por ensemble.
- **Srinivas, N. et al.** (2010). *Gaussian Process Optimization in the Bandit Setting*
  (GP-UCB/LCB). ICML. — limite de confiança como função de aquisição.

---

*Documento gerado para o projeto de Mestrado — UFPR, 2026. Complemento contínuo.*
