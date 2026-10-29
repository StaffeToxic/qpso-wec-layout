# Tabela de comparacao: artigo vs PSO/QPSO do codigo

Fontes reais usadas:

- Artigo `3377930.3390235 (1).pdf`, Tabela 1, apenas cenarios de 49 boias.
- Artigo `3377930.3390235 (1).pdf`, Figura 6, q-factor dos melhores layouts de 49 boias.
- `comparar_pso_artigo_qpso.py`, rodado em 10 sementes nesta geracao da tabela.
- Bancos `WEC_Sydney_49.db` e `WEC_Perth_49.db`, coluna `qW` para o q-factor do melhor layout presente no dataset.

Observacao importante: os valores do codigo sao avaliados pelo RandomForest surrogate. Os valores do artigo sao potencia do artigo. A comparacao e util como referencia, mas nao substitui uma reavaliacao fisica/BEM dos layouts gerados pelo codigo.

## Tabela principal

| Dataset | Origem | Metodo | Runs | Max/Melhor (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | q-factor | Tempo medio (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sydney 49 | Artigo Tabela 1 | bPSO | 10 | 4,036,728 | 3,992,467 | 4,015,029 | 4,015,661 | 11,643 | N/R | N/R |
| Sydney 49 | Artigo Tabela 1/Figura 6 | MS-bGA | 10 | 4,145,252 | 4,106,192 | 4,125,113 | 4,127,949 | 12,113 | 0.80 | N/R |
| Sydney 49 | Nosso codigo RF surrogate | PSO adaptado bPSO | 10 | 4,099,175.55 | 4,093,959.80 | 4,096,681.21 | 4,096,465.24 | 1,594.47 | N/G | 2.93 |
| Sydney 49 | Nosso codigo RF surrogate | QPSO puro | 10 | 4,145,252.22 | 4,145,252.22 | 4,145,252.22 | 4,145,252.22 | 0 | 0.80 | 2.88 |
| Perth 49 | Artigo Tabela 1 | bPSO | 10 | 3,951,612 | 3,889,598 | 3,912,453 | 3,912,960 | 17,177 | N/R | N/R |
| Perth 49 | Artigo Tabela 1/Figura 6 | MS-bDE | 10 | 4,177,659 | 4,108,205 | 4,101,260 | 4,095,057 | 33,245 | 0.88 | N/R |
| Perth 49 | Nosso codigo RF surrogate | PSO adaptado bPSO | 10 | 4,122,243.12 | 4,113,247.04 | 4,117,599.77 | 4,117,857.29 | 2,675.64 | N/G | 3.07 |
| Perth 49 | Nosso codigo RF surrogate | QPSO puro | 10 | 4,177,658.97 | 4,177,658.97 | 4,177,658.97 | 4,177,658.97 | 0 | 0.88 | 3.00 |

Legenda: N/R = nao reportado pelo artigo nessa tabela; N/G = nao gerado pelo codigo atual.

## Todos os metodos do artigo para 49 boias

### Sydney 49

| Metodo artigo | Max (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | q-factor |
|---|---:|---:|---:|---:|---:|---:|
| 1+1EA | 4,082,524 | 4,046,311 | 4,062,828 | 4,060,505 | 9,580 | N/R |
| 2+2CMAES | 4,036,152 | 4,013,471 | 4,026,230 | 4,029,230 | 7,808 | N/R |
| CMAES | 4,061,528 | 4,015,919 | 4,029,509 | 4,030,342 | 12,500 | N/R |
| DE | 4,028,337 | 4,002,550 | 4,014,852 | 4,015,482 | 8,591 | N/R |
| IDE | 4,062,984 | 3,976,301 | 4,014,611 | 4,011,081 | 20,268 | N/R |
| LS-NM | 4,089,731 | 4,039,525 | 4,063,963 | 4,064,273 | 18,106 | N/R |
| bGA | 4,052,984 | 4,006,283 | 4,028,394 | 4,025,310 | 13,725 | N/R |
| bDE | 4,078,389 | 4,050,349 | 4,062,048 | 4,059,954 | 7,924 | N/R |
| bPSO | 4,036,728 | 3,992,467 | 4,015,029 | 4,015,661 | 11,643 | N/R |
| DLS(I) | 4,108,751 | 4,070,321 | 4,092,373 | 4,095,077 | 10,140 | N/R |
| DLS(II) | 4,134,622 | 4,098,276 | 4,116,149 | 4,113,040 | 10,814 | N/R |
| SLSNM-bGA | 4,105,401 | 4,031,291 | 4,075,606 | 4,072,725 | 19,897 | N/R |
| SLSNM-bDE | 4,100,089 | 4,047,364 | 4,082,176 | 4,083,311 | 13,306 | N/R |
| SLSNM-bPSO | 4,084,955 | 4,066,979 | 4,076,665 | 4,076,621 | 4,946 | N/R |
| MS-bGA | 4,145,252 | 4,106,192 | 4,125,113 | 4,127,949 | 12,113 | 0.80 |
| MS-bDE | 4,132,385 | 4,097,087 | 4,113,961 | 4,113,672 | 9,324 | N/R |
| MS-bPSO | 4,125,968 | 4,107,365 | 4,117,115 | 4,117,134 | 5,856 | N/R |

### Perth 49

| Metodo artigo | Max (W) | Min (W) | Media (W) | Mediana (W) | Std (W) | q-factor |
|---|---:|---:|---:|---:|---:|---:|
| 1+1EA | 3,933,532 | 3,822,270 | 3,897,954 | 3,904,627 | 32,160 | N/R |
| 2+2CMAES | 3,863,159 | 3,819,043 | 3,847,996 | 3,853,485 | 14,659 | N/R |
| CMAES | 3,946,189 | 3,900,198 | 3,920,466 | 3,923,422 | 17,904 | N/R |
| DE | 3,860,444 | 3,817,360 | 3,841,577 | 3,842,720 | 13,118 | N/R |
| IDE | 3,875,783 | 3,774,422 | 3,841,125 | 3,848,962 | 26,134 | N/R |
| LS-NM | 4,080,796 | 3,870,593 | 4,001,900 | 4,008,882 | 57,673 | N/R |
| bGA | 3,984,477 | 3,884,613 | 3,925,487 | 3,919,409 | 30,027 | N/R |
| bDE | 4,010,788 | 3,914,666 | 3,960,060 | 3,964,299 | 27,647 | N/R |
| bPSO | 3,951,612 | 3,889,598 | 3,912,453 | 3,912,960 | 17,177 | N/R |
| DLS(I) | 4,029,731 | 3,916,594 | 3,984,165 | 3,987,689 | 33,047 | N/R |
| DLS(II) | 4,075,129 | 4,034,768 | 4,045,675 | 4,044,025 | 10,639 | N/R |
| SLSNM-bGA | 4,112,549 | 4,071,398 | 4,092,000 | 4,090,929 | 10,535 | N/R |
| SLSNM-bDE | 4,097,756 | 4,055,033 | 4,080,173 | 4,081,414 | 14,291 | N/R |
| SLSNM-bPSO | 4,112,602 | 4,065,949 | 4,082,473 | 4,082,183 | 12,556 | N/R |
| MS-bGA | 4,143,453 | 4,080,411 | 4,101,887 | 4,099,826 | 15,447 | N/R |
| MS-bDE | 4,177,659 | 4,108,205 | 4,101,260 | 4,095,057 | 33,245 | 0.88 |
| MS-bPSO | 4,153,355 | 4,081,240 | 4,095,598 | 4,089,500 | 19,794 | N/R |

## Leitura direta

- Sydney 49: nosso PSO adaptado teve maximo 4,099,175.55 W; bPSO do artigo teve maximo 4,036,728 W; melhor metodo do artigo (MS-bGA) teve maximo 4,145,252 W; QPSO puro teve maximo 4,145,252.22 W.
- Perth 49: nosso PSO adaptado teve maximo 4,122,243.12 W; bPSO do artigo teve maximo 3,951,612 W; melhor metodo do artigo (MS-bDE) teve maximo 4,177,659 W; QPSO puro teve maximo 4,177,658.97 W.

Nota: para o PSO adaptado, o q-factor nao foi gerado porque o resultado salvo do codigo e uma potencia predita pelo RF e o arquivo JSON nao armazena uma avaliacao fisica com `qW` do layout final. Para o QPSO puro, o layout final coincide com o melhor layout do banco, por isso o `qW` do banco foi usado.
