# Base teorica PSO para comparacao com QPSO

Este arquivo consolida a teoria do PSO que deve ser usada como base de
comparacao contra o modelo quantico/QPSO do projeto. Ele separa tres coisas que
nao devem ser misturadas:

1. PSO classico continuo, que e a formulacao teorica padrao.
2. bPSO do artigo base, que e a versao binaria/discreta usada como referencia
   para layouts em grade.
3. QPSO continuo do projeto, que e o modelo quantico/inspirado em mecanica
   quantica usado para comparar desempenho.

## Veredito da auditoria

A teoria do PSO classico em `teoria_qpso_hibrido_ml.md` esta correta na parte
central: o algoritmo usa particulas com posicao, velocidade, melhor posicao
individual (`pbest`) e melhor posicao global (`gbest`). A atualizacao de
velocidade combina inercia, atracao cognitiva e atracao social.

O ponto que precisava ficar mais claro e que o artigo base nao usa exatamente o
PSO continuo generico como baseline principal. Ele descreve uma variante
binaria, o bPSO, para selecionar celulas ativas de uma grade. Por isso, a base
mais fiel ao artigo para comparacao com o QPSO do projeto e o bPSO adaptado em
`comparar_pso_artigo_qpso.py`, nao apenas a equacao continua do PSO.

O arquivo `base_teorica.txt` nao esta organizado como uma base teorica limpa de
PSO para comparacao. Para a comparacao tecnica, use este documento como
referencia de leitura e use `comparar_pso_artigo_qpso.py` como baseline
executavel.

## Metrica correta do artigo

A metrica de otimizacao do artigo base nao e MAE. O artigo otimiza potencia
produzida pelo layout, representada como `P_sum(x, y)` ou potencia anual media,
dependendo do trecho. O artigo tambem usa o `q-factor` como metrica de qualidade
do layout.

No projeto, quando usamos RandomForest como surrogate, a funcao objetivo passa a
ser a potencia predita pelo modelo:

```text
maximize P_hat(layout)
```

Isso e uma comparacao de otimizadores sobre o surrogate, nao uma validacao
fisica/BEM completa. Portanto, nos relatorios e graficos, o correto e chamar o
ganho de ganho predito pelo RF ou ganho no surrogate.

## PSO classico continuo

No PSO classico, cada particula `i` possui:

```text
x_i^t = posicao da particula na iteracao t
v_i^t = velocidade da particula na iteracao t
p_i   = melhor posicao historica da particula, ou pbest
g     = melhor posicao global do enxame, ou gbest
```

A atualizacao padrao e:

```text
v_i^(t+1) = omega * v_i^t
          + c1 * r1 * (p_i - x_i^t)
          + c2 * r2 * (g - x_i^t)

x_i^(t+1) = x_i^t + v_i^(t+1)
```

Onde:

- `omega` controla a inercia da particula.
- `c1` controla o termo cognitivo, isto e, a tendencia de voltar ao proprio
  melhor ponto.
- `c2` controla o termo social, isto e, a tendencia de seguir o melhor ponto do
  enxame.
- `r1` e `r2` sao numeros aleatorios uniformes em `[0, 1]`.

Essa e a base teorica correta para explicar PSO. Contudo, ela nao e a base mais
fiel para reproduzir o artigo, porque o artigo usa uma representacao binaria.

## bPSO do artigo base

O artigo usa Enhanced binary PSO como uma das tecnicas avaliadas. A parte central
do bPSO e:

```text
v_i^(t+1) = omega * v_i^t
          + c1 * r1 * (p_i - x_i^t)
          + c2 * r2 * (g - x_i^t)
```

Mas a posicao nao e atualizada por soma continua. Como a particula e binaria, a
velocidade e convertida em probabilidade de inverter o bit por uma funcao de
transferencia em V:

```text
T(v) = abs((2 / pi) * arctan((pi / 2) * v))
```

A atualizacao do bit e:

```text
if rand < T(v_i^(t+1)):
    x_i^(t+1) = 1 - x_i^t
else:
    x_i^(t+1) = x_i^t
```

No artigo, os parametros extraidos para o bPSO sao:

```text
lambda = 12
c1 = 2
c2 = 2
omega_max = 0.9
omega_min = 0.4
```

O texto extraido do PDF fica ambiguo ao redor de `omega_ini = 2`. Na
implementacao do projeto, isso foi tratado como escala inicial de velocidade
igual a `2.0`, mantendo a inercia com decaimento linear de `0.9` para `0.4`.
Essa e a interpretacao mais coerente com `omega_max` e `omega_min`.

## Baseline executavel no projeto

O baseline PSO do artigo esta implementado em:

```text
comparar_pso_artigo_qpso.py
```

Classe principal:

```text
ArticleBPSOGrid_WEC
```

O que essa implementacao preserva do artigo:

- Populacao `lambda = 12`.
- Coeficientes `c1 = 2` e `c2 = 2`.
- Inercia com decaimento linear.
- Funcao de transferencia em V.
- Atualizacao binaria por inversao de bit.
- Grade discreta com passo de `50 m`.
- Reparo para manter exatamente `49` WECs ativos.

O que essa implementacao nao reproduz integralmente:

- Inicializacao esperta SLS-NM completa do artigo.
- Estrategias DLS, CLS, rotacao e variantes hibridas completas.
- Modelo hidrodinamico/fisico completo do artigo.
- Todas as configuracoes experimentais originais do benchmark.

Portanto, esse arquivo deve ser chamado de:

```text
bPSO do artigo adaptado ao surrogate do projeto
```

Nao chame de reproducao completa do artigo.

## QPSO continuo do projeto

O QPSO puro do projeto esta implementado em:

```text
qpso_layout_puro_standalone.py
```

Classe principal:

```text
PureContinuousQPSO_WEC
```

No QPSO, nao existe vetor de velocidade classico. A particula e atualizada por
um atrator local e por um termo probabilistico inspirado no modelo quantico:

```text
p_i = phi * pbest_i + (1 - phi) * gbest
mbest = mean(pbest)

x_i^(t+1) = p_i +/- alpha * abs(mbest - x_i^t) * ln(1 / u)
```

Onde:

- `phi` e aleatorio em `[0, 1]`.
- `u` e aleatorio em `(0, 1]`.
- o sinal `+/-` e aleatorio.
- `alpha` controla a contracao-expansao da busca.
- `mbest` e a media dos melhores historicos individuais.

No projeto, o QPSO puro usa:

```text
alpha_start = 1.0
alpha_end = 0.5
```

Isso e adequado para uma comparacao empirica, mas nao deve ser descrito como
prova formal de convergencia global. E uma estrategia pratica de contracao da
busca.

## PSO vs QPSO: diferenca teorica central

| Aspecto | PSO classico | bPSO do artigo | QPSO puro |
|---|---|---|---|
| Representacao | Continua | Binaria/discreta | Continua |
| Estado principal | Posicao e velocidade | Bits e velocidade auxiliar | Posicao, pbest, gbest e mbest |
| Movimento | Soma da velocidade | Probabilidade de inverter bits | Amostragem ao redor do atrator |
| Parametros principais | omega, c1, c2 | omega, c1, c2, transferencia V | alpha, phi, mbest |
| Base para o artigo | Teorica geral | Sim, baseline mais fiel | Nao, e o modelo comparado |
| Comparacao justa com QPSO continuo | Media | Parcial, pois muda a representacao | Referencia quantica |

## Como comparar corretamente

Existem dois tipos de comparacao, e eles precisam ser separados.

### 1. Comparacao com o artigo

Use:

```text
ArticleBPSOGrid_WEC vs PureContinuousQPSO_WEC
```

Essa comparacao responde:

```text
Meu QPSO supera o bPSO inspirado no artigo quando ambos sao avaliados pelo
surrogate RandomForest do projeto?
```

Cuidado: como o bPSO e discreto e o QPSO e continuo, essa comparacao e fiel ao
artigo, mas nao e perfeitamente "algoritmo contra algoritmo" em mesma
representacao.

### 2. Comparacao algoritmica justa

Para uma comparacao mais justa entre algoritmos, o ideal e usar:

```text
Continuous PSO vs PureContinuousQPSO
```

Essa comparacao responderia:

```text
A parte quantica do QPSO melhora a busca quando a representacao, os limites, o
modelo surrogate e o orcamento computacional sao iguais?
```

Essa segunda comparacao ainda nao substitui o bPSO do artigo, mas e melhor para
isolar o efeito do modelo quantico.

## Protocolo recomendado

Para qualquer resultado comparativo, usar:

- Mesmos datasets: Sydney e Perth.
- Mesmo modelo surrogate RandomForest.
- Mesma funcao objetivo: potencia predita pelo RF.
- Mesmo numero de WECs.
- Mesmo limite de iteracoes.
- Mesmo numero de avaliacoes sempre que possivel.
- Varias seeds independentes.
- Relatar melhor, media, mediana, desvio padrao e tempo.
- Mostrar curva de convergencia.
- Separar resultado surrogate de validacao fisica.

Formato minimo recomendado:

```text
Dataset | Algoritmo | Melhor RF | Media RF | Desvio | Tempo | Seed
```

## Interpretacao dos resultados atuais

Os arquivos atuais permitem uma comparacao inicial entre:

```text
comparar_pso_artigo_qpso.py
qpso_layout_puro_standalone.py
```

Essa comparacao deve ser interpretada assim:

- Se o QPSO ganha do bPSO, ele ganhou dentro do surrogate RandomForest do
  projeto.
- Se o QPSO fica proximo dos melhores valores do artigo, isso nao prova sozinho
  que ele superou o artigo, porque a avaliacao fisica e a metodologia completa
  do artigo nao foram reproduzidas.
- Se o QPSO usa o melhor layout conhecido como ponto inicial, o resultado mostra
  refinamento/preservacao de uma boa solucao, nao descoberta independente desde
  uma populacao totalmente aleatoria.

## Conclusao para usar na monografia

O PSO deve ser apresentado como a base classica de enxame, em que as particulas
se movem por uma velocidade influenciada por inercia, memoria individual e
memoria social. Para comparacao com o artigo, a referencia correta e o bPSO com
funcao de transferencia em V, pois o artigo trabalha com uma codificacao binaria
de layouts em grade.

O QPSO deve ser apresentado como uma extensao quantica/inspirada em mecanica
quantica que remove a velocidade explicita e usa um atrator probabilistico
baseado em `pbest`, `gbest` e `mbest`. Assim, a comparacao fica teoricamente
organizada:

```text
PSO classico: explica a base.
bPSO do artigo: baseline fiel ao artigo.
QPSO puro: modelo quantico proposto/comparado.
```

Para resultados numericos, usar `comparar_pso_artigo_qpso.py` como baseline
principal e declarar explicitamente que os valores sao avaliados pelo surrogate
RandomForest do projeto.
