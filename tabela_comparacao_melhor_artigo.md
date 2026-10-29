# Comparacao contra o melhor metodo do artigo

Baseline usado em cada dataset:

- Sydney 49: melhor do artigo = `MS-bGA`, com `Max = 4,145,252 W`.
- Perth 49: melhor do artigo = `MS-bDE`, com `Max = 4,177,659 W`.

Formula da diferenca percentual:

`Dif. % vs melhor artigo = (Max metodo - Max melhor artigo) / Max melhor artigo * 100`

| Dataset | Metodo | Max/Melhor P_sum (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | Dif. W vs melhor artigo | Dif. % vs melhor artigo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sydney 49 | Melhor artigo (MS-bGA) | 4,145,252 | 4,106,192 | 4,125,113 | 4,127,949 | 12,113 | +0.00 | +0.00% |
| Sydney 49 | bPSO do artigo | 4,036,728 | 3,992,467 | 4,015,029 | 4,015,661 | 11,643 | -108,524.00 | -2.62% |
| Sydney 49 | MS-bPSO do artigo | 4,125,968 | 4,107,365 | 4,117,115 | 4,117,134 | 5,856 | -19,284.00 | -0.47% |
| Sydney 49 | QPSO standalone layout | 4,139,138.06 | 4,138,660.34 | 4,138,914.46 | 4,138,971.54 | 153.04 | -6,113.94 | -0.15% |
| Perth 49 | Melhor artigo (MS-bDE) | 4,177,659 | 4,108,205 | 4,101,260 | 4,095,057 | 33,245 | +0.00 | +0.00% |
| Perth 49 | bPSO do artigo | 3,951,612 | 3,889,598 | 3,912,453 | 3,912,960 | 17,177 | -226,047.00 | -5.41% |
| Perth 49 | MS-bPSO do artigo | 4,153,355 | 4,081,240 | 4,095,598 | 4,089,500 | 19,794 | -24,304.00 | -0.58% |
| Perth 49 | QPSO standalone layout | 4,177,658.97 | 4,175,268.20 | 4,177,400.27 | 4,177,658.97 | 751.67 | -0.03* | -0.00% |

Leitura direta:

- No Sydney 49, o melhor do artigo foi `MS-bGA`. O `QPSO standalone layout` ficou `0.15%` abaixo.
- No Perth 49, o melhor do artigo foi `MS-bDE`. O `QPSO standalone layout` empatou com ele dentro do arredondamento.
- O `MS-bPSO` fica abaixo do melhor artigo nos dois datasets: `-0.47%` em Sydney 49 e `-0.58%` em Perth 49.
- O `bPSO` fica bem abaixo do melhor artigo: `-2.62%` em Sydney 49 e `-5.41%` em Perth 49.

Legenda:

- `*` = diferenca causada por arredondamento entre o valor inteiro reportado no artigo e o valor decimal salvo no dataset/codigo; deve ser interpretado como empate pratico.

Observacao: os resultados do artigo foram retirados da Tabela 1. Os resultados
do nosso `QPSO standalone layout` foram retirados dos resultados salvos pelo
codigo local.
