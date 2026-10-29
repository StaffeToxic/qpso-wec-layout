# Relatório Comparativo: Algoritmos Híbridos de Inspiração Quântica

**[Atualizado em 06/04/2026 — Execução exclusiva de algoritmos HÍBRIDOS × Sydney 49 WECs]**

---

## Contextualização

Este relatório compara exclusivamente os **algoritmos híbridos** (Quântico + Genético) implementados para o problema de seleção de boias WEC. Todos utilizam o dataset `WEC_Sydney_49.db` (17.964 layouts, 49 boias, R² = 0.9815).

A potência de referência com todas as 49 boias ativas é **4.145.252,22 W**.

---

## Parte 1 — Algoritmos Híbridos Base (60 iterações, 25 indivíduos)

| # | Algoritmo | Componente Quântico | Componente Genético | Fitness | Boias | Redução |
|---|:---|:---|:---|:---|:---|:---|
| 1 | QAOA-Híbrido | Fases γ/β (Problema+Mixing) | Mutação GA (p=0.30) | **3.942.292,61** | 26/49 | 46,9% |
| 2 | QEA-Híbrido | Rotação de QuBits (θ=0.05π) | Mutação GA (p=0.30) | **3.944.270,29** | 33/49 | 32,7% |
| 3 | QPSO-Híbrido | Poço Delta Schrödinger + Sigmoid | Mutação GA (p=0.30) | **3.941.984,31** | 32/49 | 34,7% |

**Vencedor Base: QEA-Híbrido** — Maior fitness (3.944.270) com redução significativa (33 boias).

---

## Parte 2 — Algoritmos Híbridos Enhanced (80 iterações, 30 indivíduos)

| # | Algoritmo | Melhorias Enhanced | Fitness | Boias | Redução | Tempo |
|---|:---|:---|:---|:---|:---|:---|
| 1 | QAOA-Enhanced | Elitismo, Busca Local ×10, Mutação Adaptativa, Re-diversificação | **3.944.237,56** | 21/49 | **57,1%** | 520,2s |
| 2 | QEA-Enhanced | Rotação 3 níveis, Perturbação Quântica, Injeção de Elite | **3.944.458,13** | 30/49 | **38,8%** | 137,8s |
| 3 | QPSO-Enhanced | Multi-Strategy, Lévy Flight, Crossover gbest, Reparação Inteligente | **3.942.388,74** | 39/49 | 20,4% | 138,1s |

**Vencedor Enhanced: QEA-Enhanced** — Fitness máximo absoluto (3.944.458) com boa redução (30 boias).

---

## Parte 3 — QPSO Híbrido Discreto Standalone (80 iter, 30 partículas, min_boias=25)

| Métrica | Valor |
|:---|:---|
| R² Score (RF 200 árvores) | 0.9804 |
| Potência Original (49/49 boias) | 4.145.252,22 W |
| Potência Otimizada (48/49 boias) | 4.145.226,01 W |
| Perda de potência | **0,001%** |
| Boia removida | **#26** |
| Eficiência por boia (original) | 84.596,98 W/boia |
| Eficiência por boia (otimizada) | 86.358,88 W/boia |
| Ganho de eficiência | **+2,08%** |

> O QPSO Standalone com `min_boias=25` foi extremamente conservador, removendo apenas 1 boia (#26), que contribui marginalmente à potência total.

---

## Parte 4 — Comparação Base vs Enhanced

| Algoritmo | Fitness Base → Enhanced | Δ Fitness | Boias Base → Enhanced | Δ Boias |
|:---|:---|:---|:---|:---|
| QAOA | 3.942.292 → 3.944.237 | **+1.945** | 26 → 21 | **−5** |
| QEA | 3.944.270 → 3.944.458 | **+188** | 33 → 30 | **−3** |
| QPSO | 3.941.984 → 3.942.388 | **+404** | 32 → 39 | **+7** |

### Observações:
- **QAOA** se beneficiou enormemente da mutação adaptativa + elitismo: ganhou ~2.000 pontos de fitness e reduziu 5 boias
- **QEA** já era forte na versão base; a versão Enhanced refinou marginalmente (+188) e cortou 3 boias
- **QPSO Enhanced** ficou mais conservador que a versão base (39 vs 32 boias) — o crossover com gbest tende a "copiar" bits ativos

---

## Análise: Qual Algoritmo Híbrido Usar?

| Cenário | Algoritmo Recomendado | Justificativa |
|:---|:---|:---|
| Melhor fitness absoluto | **QEA-Enhanced** | Fitness máximo: 3.944.458 |
| Redução máxima de custos | **QAOA-Enhanced** | Apenas 21/49 boias (−57%) |
| Máxima preservação de potência | **QPSO Híbrido Discreto** | 48/49 boias com ≈0% perda |
| Equilíbrio geral (fitness + redução) | **QEA-Enhanced** | 30/49 boias com fitness mais alto |
| Velocidade de execução | **QEA-Enhanced ou QPSO-Enhanced** | ~138s vs 520s do QAOA |

---

## Por Que o QEA-Enhanced Vence?

1. **Rotação quântica direcionada**: A rotação de qubits guiada pelo gbest converge mais rápido que a busca estocástica do QPSO ou a busca local do QAOA.
2. **Perturbação quântica**: O ruído de 2% nos ângulos θ evita convergência prematura sem comprometer a trajetória.
3. **Mutação adaptativa**: A taxa decrescente (0.35→0.05) maximiza exploração inicial e refinamento final.
4. **Injeção periódica de elite**: A cada 20 iterações, um indivíduo recebe o "template" do gbest nos seus qubits, acelerando a propagação de boas soluções.

---

## Conclusão do Comparativo

> A hibridização quântica + genética é consistentemente superior às versões base em TODOS os algoritmos.
>
> O **QEA-Enhanced** é o algoritmo recomendado para otimização geral de WECs:
> - **Fitness: 3.944.458** (máximo entre todos)
> - **30/49 boias** (redução de 38,8% nos equipamentos)
> - **137,8 segundos** de execução
>
> Para cenários de redução radical, o **QAOA-Enhanced** com 21/49 boias oferece a maior economia de infraestrutura.

---

*Fonte: Execução de `algoritmos_hibridos_wec.py`, `algoritmos_hibridos_enhanced.py` e `qpso_hibrido_discreto_wec.py` em 06/04/2026 sobre WEC_Sydney_49.db*
