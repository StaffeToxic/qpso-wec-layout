# Comparacao com o artigo base

Fonte:

- Artigo `3377930.3390235 (1).pdf`, Tabela 1
- `comparacao_pso_artigo_qpso.json`
- `resultado_qpso_layout_puro.json`

Escopo:

- Apenas cenarios de 49 boias, porque os scripts atuais usam `WEC_Sydney_49.db` e `WEC_Perth_49.db`.
- A metrica e potencia total em W.
- Os numeros do nosso codigo sao predicoes do RandomForest surrogate.

## 1. Resultados do artigo para 49 boias

| Dataset | bPSO artigo Max | bPSO artigo Mean | Melhor metodo do artigo | Melhor Max artigo |
|---|---:|---:|---|---:|
| Sydney 49 | 4,036,728 W | 4,015,029 W | MS-bGA | 4,145,252 W |
| Perth 49 | 3,951,612 W | 3,912,453 W | MS-bDE | 4,177,659 W |

## 2. Nossos resultados

| Dataset | Nosso PSO adaptado | Meu QPSO puro |
|---|---:|---:|
| Sydney 49 | 4,096,998.89 W | 4,145,252.22 W |
| Perth 49 | 4,114,256.36 W | 4,177,658.97 W |

## 3. Comparacao direta contra o bPSO do artigo

| Dataset | Nosso PSO vs bPSO artigo Max | Meu QPSO vs bPSO artigo Max |
|---|---:|---:|
| Sydney 49 | +60,270.89 W (+1.49%) | +108,524.22 W (+2.69%) |
| Perth 49 | +162,644.36 W (+4.12%) | +226,046.97 W (+5.72%) |

## 4. Comparacao contra o melhor metodo do artigo

| Dataset | Melhor metodo do artigo | Nosso PSO adaptado | Meu QPSO puro |
|---|---:|---:|---:|
| Sydney 49 | MS-bGA: 4,145,252 W | -48,253.11 W (-1.16%) | +0.22 W (+0.00%) |
| Perth 49 | MS-bDE: 4,177,659 W | -63,402.64 W (-1.52%) | -0.03 W (-0.00%) |

## 5. Leitura correta

Em relacao ao bPSO isolado do artigo, os dois resultados do nosso codigo ficaram melhores:

- O PSO adaptado ficou acima do bPSO do artigo em Sydney e Perth.
- O QPSO puro ficou ainda acima do bPSO do artigo em Sydney e Perth.

Em relacao ao melhor resultado global do artigo para 49 boias:

- O QPSO puro praticamente empata com o melhor resultado do artigo.
- O PSO adaptado fica abaixo do melhor metodo do artigo.

A interpretacao mais honesta e que o QPSO puro preservou o melhor layout continuo ja existente no dataset. Por isso ele aparece praticamente igual ao melhor valor do artigo. Isso nao significa, sozinho, que o QPSO descobriu um layout novo melhor que o artigo; significa que, nesta configuracao, ele manteve o melhor layout de referencia enquanto o PSO em grade discreta perdeu um pouco de potencia.

## 6. Frase pronta para o trabalho

Comparando com os resultados do artigo para 49 boias, o QPSO puro alcancou potencia predita equivalente ao melhor resultado reportado no artigo nos cenarios Sydney e Perth, enquanto o PSO binario adaptado ficou acima do bPSO isolado do artigo, mas abaixo dos melhores metodos multi-estrategia do artigo. A comparacao deve ser interpretada com cautela, pois os resultados do codigo atual sao obtidos por RandomForest surrogate e o QPSO preserva o melhor layout continuo de referencia do dataset.

> **Nota de cautela (investigacao posterior).** A "equivalencia ao melhor do artigo" reflete o fato de o QPSO **preservar** o melhor layout de referencia (o RF, constante por partes, nao tem o que melhorar), e nao uma descoberta de layout novo superior. Um experimento posterior com surrogate suave + cross-check (ver `teoria_qpso_continuo_layout.md` §8) confirmou que ganhos continuos acima do melhor layout **nao se sustentam** sob validacao cruzada. Portanto, NAO afirmar no trabalho que o metodo "supera" o artigo; afirmar que **iguala/preserva** o melhor layout de referencia.
