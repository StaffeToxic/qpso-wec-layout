# Tabela: bPSO do artigo vs QPSO puro vs meu QPSO

Fontes usadas:

- Artigo `3377930.3390235 (1).pdf`, Tabela 1, cenarios de 49 boias.
- `tabela_comparacao_artigo_codigo_metricas.json`, gerado com 10 sementes para o QPSO puro.
- Execucao atual de `qpso_layout_standalone.py` em 2026-05-08 para o meu QPSO layout standalone.

Observacao: os valores do nosso codigo sao potencia predita pelo RandomForest
surrogate. Os valores do artigo sao os valores de potencia reportados no artigo.

| Dataset | Metodo | Natureza | Runs | Max/Melhor (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | q-factor | Tempo medio (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sydney 49 | bPSO do artigo | PSO binario/discreto do artigo | 10 | 4,036,728 | 3,992,467 | 4,015,029 | 4,015,661 | 11,643 | N/R | N/R |
| Sydney 49 | QPSO puro | QPSO continuo sem Levy, elitismo ou crossover | 10 | 4,145,252.22 | 4,145,252.22 | 4,145,252.22 | 4,145,252.22 | 0 | 0.80 | 2.88 |
| Sydney 49 | Meu QPSO layout standalone | QPSO continuo hibrido com inicializacao por dataset, elitismo e Levy | 1 | 4,138,536.30 | 4,138,536.30 | 4,138,536.30 | 4,138,536.30 | N/A | N/G | 9.80 |
| Perth 49 | bPSO do artigo | PSO binario/discreto do artigo | 10 | 3,951,612 | 3,889,598 | 3,912,453 | 3,912,960 | 17,177 | N/R | N/R |
| Perth 49 | QPSO puro | QPSO continuo sem Levy, elitismo ou crossover | 10 | 4,177,658.97 | 4,177,658.97 | 4,177,658.97 | 4,177,658.97 | 0 | 0.88 | 3.00 |
| Perth 49 | Meu QPSO layout standalone | QPSO continuo hibrido com inicializacao por dataset, elitismo e Levy | 1 | 4,177,658.97 | 4,177,658.97 | 4,177,658.97 | 4,177,658.97 | N/A | N/G | 9.80 |

Legenda:

- N/R = nao reportado pelo artigo.
- N/G = nao gerado pelo codigo atual.
- N/A = nao aplicavel para uma unica execucao.

## Leitura curta

Em Sydney 49, o QPSO puro atingiu o melhor valor do artigo para 49 boias
(`4,145,252 W`), enquanto o meu QPSO layout standalone ficou um pouco abaixo
(`4,138,536.30 W`). O bPSO isolado do artigo ficou abaixo dos dois.

Em Perth 49, tanto o QPSO puro quanto o meu QPSO layout standalone atingiram
`4,177,658.97 W`, praticamente o mesmo valor maximo reportado pelo melhor metodo
do artigo para 49 boias (`4,177,659 W`). O bPSO isolado do artigo ficou abaixo.

O QPSO puro e a comparacao teorica mais limpa. O meu QPSO layout standalone deve
ser apresentado como versao hibrida/enhanced, porque inclui elitismo, voos de
Levy e inicializacao baseada em layouts do dataset.
