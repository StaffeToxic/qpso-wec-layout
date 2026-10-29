# Tabela comparativa com metricas do artigo

Metricas usadas: potencia total `P_sum(x, y)` em W, minimo, media, mediana,
maximo e desvio padrao. A porcentagem final usa o `bPSO` do artigo como
baseline em cada dataset:

`Dif. % vs bPSO = (Max metodo - Max bPSO) / Max bPSO * 100`

| Dataset | Metodo | Max/Melhor P_sum (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | Dif. % vs bPSO |
|---|---|---:|---:|---:|---:|---:|---:|
| Sydney 49 | bPSO do artigo | 4,036,728 | 3,992,467 | 4,015,029 | 4,015,661 | 11,643 | +0.00% |
| Sydney 49 | MS-bPSO do artigo | 4,125,968 | 4,107,365 | 4,117,115 | 4,117,134 | 5,856 | +2.21% |
| Sydney 49 | QPSO standalone layout | 4,139,165.22 | 4,138,423.49 | 4,138,894.63 | 4,138,923.10 | 226.99 | +2.54% |
| Perth 49 | bPSO do artigo | 3,951,612 | 3,889,598 | 3,912,453 | 3,912,960 | 17,177 | +0.00% |
| Perth 49 | MS-bPSO do artigo | 4,153,355 | 4,081,240 | 4,095,598 | 4,089,500 | 19,794 | +5.11% |
| Perth 49 | QPSO standalone layout | 4,177,658.97 | 4,175,268.20 | 4,177,419.89 | 4,177,658.97 | 756.03 | +5.72% |

Leitura direta:

- O MS-bPSO melhora o bPSO nos dois datasets.
- O QPSO standalone layout fica acima do bPSO nos dois datasets.

Observacao: os resultados do artigo sao os valores reportados na Tabela 1. Os
resultados do nosso QPSO standalone layout foram obtidos pelo codigo local com
avaliacao via surrogate RandomForest; portanto, a comparacao e util como
experimento computacional interno, mas nao substitui uma reavaliacao fisica/BEM
dos layouts finais.
