# Comparacao entre meu QPSO e o PSO do artigo

Fonte dos resultados:

- `comparacao_pso_artigo_qpso.json`
- `resultado_qpso_layout_puro.json`
- Execucao em `WEC_Sydney_49.db` e `WEC_Perth_49.db`
- Metrica comparada: potencia total predita pelo RandomForest surrogate, em W

## 1. Diferenca entre os algoritmos

| Criterio | Meu QPSO puro | PSO do artigo |
|---|---|---|
| Arquivo | `qpso_layout_puro_standalone.py` | `comparar_pso_artigo_qpso.py` |
| Classe | `PureContinuousQPSO_WEC` | `ArticleBPSOGrid_WEC` |
| Tipo de busca | Continua | Binaria/discreta |
| Variavel otimizada | Coordenadas X/Y diretamente | Bits que ativam pontos de uma grade |
| Representacao do layout | Vetor continuo `[X1,Y1,...,X49,Y49]` | Grade discreta com pontos espacados |
| Atualizacao principal | QPSO com `pbest`, `gbest`, `mbest`, `alpha` e `ln(1/u)` | bPSO com velocidade, `c1=c2=2`, inercia e transferencia V-shaped |
| Operadores extras | Nao usa elitismo, Levy nem crossover | Nao usa Levy/crossover; usa dinamica classica de PSO binario |
| Metrica de fitness | Potencia RF predita | Potencia RF predita |

## 2. Resultados numericos

| Dataset | PSO artigo (W) | Meu QPSO puro (W) | Diferenca QPSO - PSO | Diferenca (%) |
|---|---:|---:|---:|---:|
| Sydney 49 | 4,096,998.89 | 4,145,252.22 | +48,253.33 W | +1.18% |
| Perth 49 | 4,114,256.36 | 4,177,658.97 | +63,402.61 W | +1.54% |

## 3. Comparacao contra o layout original

| Dataset | Original RF (W) | PSO artigo (W) | Meu QPSO puro (W) |
|---|---:|---:|---:|
| Sydney 49 | 4,145,252.22 | 4,096,998.89 | 4,145,252.22 |
| Perth 49 | 4,177,658.97 | 4,114,256.36 | 4,177,658.97 |

O meu QPSO puro empatou exatamente com o melhor layout original do dataset nos dois casos. Ele nao encontrou uma potencia maior, mas tambem nao degradou o melhor layout inicial.

O PSO do artigo ficou abaixo do QPSO puro nos dois datasets. Em Sydney, ficou cerca de 48.3 kW abaixo. Em Perth, cerca de 63.4 kW abaixo.

## 4. Interpretacao

Nesta execucao, o meu QPSO puro teve melhor resultado numerico que o PSO do artigo:

- Sydney 49: QPSO puro foi 1.18% melhor que o PSO.
- Perth 49: QPSO puro foi 1.54% melhor que o PSO.

Mas a leitura correta e cuidadosa e esta: o QPSO puro comeca contendo o melhor layout original continuo do dataset como referencia, entao ele consegue preservar esse valor. Ja o PSO do artigo foi adaptado como bPSO em grade discreta de 50 m; ao discretizar o espaco, ele pode perder precisao em relacao ao melhor layout continuo original.

Portanto, a vantagem do QPSO puro nesta comparacao nao prova sozinha que ele explora melhor o espaco. Ela mostra que, nesta configuracao, ele preserva melhor o melhor layout continuo conhecido, enquanto o PSO binario em grade fica limitado pela representacao discreta.

## 5. Conclusao

Para estes dois datasets e usando o mesmo RandomForest surrogate:

**Meu QPSO puro foi superior ao PSO do artigo em potencia final predita.**

No entanto, o resultado mais importante e que o QPSO puro nao melhorou o melhor layout original; ele apenas manteve esse layout como melhor solucao. Para uma comparacao mais justa de capacidade de busca, o ideal seria rodar varias sementes, remover a vantagem de inicializacao direta no melhor layout, ou aplicar ambos com a mesma representacao continua/discreta.

## 6. Atualizacao (investigacao posterior)

A causa do "empate ao centavo" foi confirmada: o **RandomForest e constante por partes**, e seu maximo sobre a caixa coincide com o melhor layout de treino — logo nao ha o que a busca continua melhore (ver `teoria_qpso_continuo_layout.md` §2). Trocando o RF por um surrogate **suave** (ensemble de MLP, R²=0,9932 > 0,9815 do RF), o QPSO **passa a otimizar de fato** (sobe ~+5% sob o MLP). Porem, um experimento multi-seed com **cross-check** mostrou que esse ganho **nao e confirmado pelo RF** (ganho RF sempre ~−1%): e extrapolacao do surrogate, nao ganho fisico. Conclusao honesta: sem ground-truth (BEM) no laco, o caso continuo nao supera de forma confiavel o melhor layout. Detalhes em `teoria_qpso_continuo_layout.md` §8–§9 e `experimentos_sydney.png`.
