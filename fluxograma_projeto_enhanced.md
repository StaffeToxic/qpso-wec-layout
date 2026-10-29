# Fluxograma — Algoritmos Híbridos Enhanced para Otimização de WECs

**Versão atualizada** — Reflete fielmente o código `algoritmos_hibridos_enhanced.py` + `shared_utils.py`

---

## 1. Visão Geral do Pipeline

```mermaid
flowchart TD
    A["🗄️ Banco de Dados\nWEC_Sydney_49.db"] --> B["📊 Extração de Features\nX1,Y1...X49,Y49 + Total_Power"]
    B --> C["🤖 Treinamento ML Surrogate\nRandomForestRegressor (150 árvores)"]
    C --> D{"Modelo Treinado\nR² ≈ 0.98\n(Prediz Potência)"}
    
    D --> E["⚛️ QAOA Enhanced"]
    D --> F["🧬 QEA Enhanced"]
    D --> G["🌊 QPSO Enhanced"]
    
    E --> H["📈 Convergência QAOA"]
    F --> I["📈 Convergência QEA"]
    G --> J["📈 Convergência QPSO"]
    
    H --> K["🏆 Comparativo Final\nMelhor Layout de 49 Boias"]
    I --> K
    J --> K
    
    K --> L["📊 Gráfico Comparativo\ncomparativo_hibridos_enhanced.png"]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#eee
    style C fill:#16213e,stroke:#0f3460,color:#eee
    style D fill:#0f3460,stroke:#53a8b6,color:#eee
    style E fill:#533483,stroke:#e94560,color:#eee
    style F fill:#2b9348,stroke:#55a630,color:#eee
    style G fill:#d62828,stroke:#f77f00,color:#eee
    style K fill:#f4a261,stroke:#e76f51,color:#1a1a2e
    style L fill:#264653,stroke:#2a9d8f,color:#eee
```

---

## 2. Função de Fitness (Compartilhada)

```mermaid
flowchart LR
    A["Vetor binário\nbits ∈ {0,1}^49"] --> B{"Σ bits\n≥ min_boias?"}
    B -- Não --> C["-1e6\n(Penalidade)"]
    B -- Sim --> D["Montar coordenadas:\nbit=1 → manter X,Y\nbit=0 → zerar X,Y"]
    D --> E["ML Surrogate\nRandomForest.predict()"]
    E --> F["potência_predita"]
    F --> G["eficiência =\npotência / n_ativas"]
    G --> H["fitness = 0.95 × potência\n+ 0.05 × eficiência"]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#eee
    style E fill:#16213e,stroke:#0f3460,color:#eee
    style H fill:#2b9348,stroke:#55a630,color:#fff
```

---

## 3. Fluxo do QAOA Enhanced

```mermaid
flowchart TD
    subgraph INIT["Inicialização"]
        A1["Gerar população aleatória\npop ∈ {0,1}^49"] --> A2["Garantir min_boias ≥ 20\npor indivíduo"]
        A3["Definir schedules:\nγ = linspace(0.9 → 0.3)\nβ = linspace(0.1 → 0.5)"]
        A4["Parâmetros Enhanced:\nelite_frac = 10%\nn_local_search = 8–10\ndiversity_threshold = 0.02"]
    end
    
    A2 --> B1

    subgraph LOOP["Loop por p camadas"]
        direction TB
        B1["🔴 FASE 1: Hamiltoniano de Custo Hc\n(Busca Local Gulosa Intensificada)"]
        B1 --> B2["Para cada amostra:\n8–10 tentativas de flip de bit\n(controlado por γ)"]
        B2 --> B3{"flip melhorou\no fitness?"}
        B3 -- Sim --> B4["Aceitar flip\nAtualizar gbest"]
        B3 -- Não --> B5["Reverter flip"]
        B4 --> B6
        B5 --> B6
        
        B6["🔵 FASE 2: Hamiltoniano Mixer Hm\n(Pauli-X Simulado)"]
        B6 --> B7["Flip aleatório com\nprob = β × 0.15"]
        
        B7 --> B7b["⭐ FASE 3: Elitismo\nSalvar top-10% da população"]
        
        B7b --> B8["🧬 FASE 4: Mutação Adaptativa\n(Operador Híbrido GA)"]
        B8 --> B9["Taxa adaptativa:\np_mut = 0.4 → 0.05\n(decaimento linear)"]
        B9 --> B9b["Inverter bits dos indivíduos\ncom fitness < mediana"]
        
        B9b --> B10["⭐ Reinserir Elite\n(top-10% preservados intactos)"]
        
        B10 --> B11{"📊 Diversidade\n< 2%?"}
        B11 -- Não --> B13["Reparar restrição\nmin_boias ≥ 20"]
        B11 -- Sim --> B12{"Fitness\nestagnou 3x?"}
        B12 -- Sim --> B12b["🔄 Re-randomizar\n1/3 da população\n+ garantir min_boias"]
        B12 -- Não --> B13
        B12b --> B13
    end
    
    B13 --> C1["Retornar gbest, fitness, histórico"]
    
    style B1 fill:#e94560,stroke:#533483,color:#fff
    style B6 fill:#0f3460,stroke:#533483,color:#fff
    style B7b fill:#f4a261,stroke:#e76f51,color:#1a1a2e
    style B8 fill:#2b9348,stroke:#55a630,color:#fff
    style B10 fill:#f4a261,stroke:#e76f51,color:#1a1a2e
    style B11 fill:#533483,stroke:#e94560,color:#fff
    style B12b fill:#d62828,stroke:#f77f00,color:#fff
    style C1 fill:#f4a261,stroke:#e76f51,color:#1a1a2e
```

---

## 4. Fluxo do QEA Enhanced

```mermaid
flowchart TD
    subgraph INIT["Inicialização"]
        A1["Criar população quântica\nq_pop[i,j] = (α,β) = (1/√2, 1/√2)"]
        A2["Superposição uniforme:\nP(0) = P(1) = 50%"]
        A3["Parâmetros Enhanced:\nrotação = medium (0.05π)\nperturbação quântica = 2%"]
    end
    
    A1 --> B1
    
    subgraph LOOP["Loop por max_iter iterações"]
        direction TB
        B1["⚛️ FASE 1: Colapso Quântico\n(Observação/Medição)"]
        B1 --> B2["Para cada qubit j:\nP(1) = |β_j|²\nSortear bit clássico"]
        B2 --> B2b["Garantir min_boias ≥ 20"]
        B2b --> B2c["Avaliar fitness\nAtualizar gbest"]
        
        B2c --> B3["🧬 FASE 2: Mutação Adaptativa\n(Híbrido GA)"]
        B3 --> B4["Taxa adaptativa:\np_mut = 0.35 → 0.05\n(decaimento linear)"]
        B4 --> B4b["Inverter bits dos piores\n(fitness < mediana)"]
        
        B4b --> B5["📊 Reavaliar fitness\nde TODOS os indivíduos\nAtualizar gbest"]
        
        B5 --> B6["🔄 FASE 3: Rotação Quântica\n(Porta Ry θ)"]
        B6 --> B7{"x_i ≠ best_i ?"}
        B7 -- "x=0, best=1" --> B8["θ = +0.05π\n(rotaciona para |1⟩)"]
        B7 -- "x=1, best=0" --> B9["θ = −0.05π\n(rotaciona para |0⟩)"]
        B7 -- "x = best" --> B10["θ = 0\n(sem rotação)"]
        
        B8 --> B10b
        B9 --> B10b
        B10 --> B10b
        
        B10b{"🎲 Perturbação\nQuântica (2%)?"}
        B10b -- Sim --> B10c["θ += ruído ∈\n[-0.02π, +0.02π]"]
        B10b -- Não --> B11
        B10c --> B11
        
        B11["Aplicar Matriz Ry:\nα' = α·cos θ − β·sin θ\nβ' = α·sin θ + β·cos θ\nNormalizar: √(α'² + β'²)"]
        
        B11 --> B12{"Iteração\nmúltiplo de 20?"}
        B12 -- Sim --> B12b["⭐ Injeção Elite:\num indivíduo recebe\nq-bits forçados do gbest\n(α=0.001/β=0.999)"]
        B12 -- Não --> B13["Próxima iteração"]
        B12b --> B13
    end
    
    B13 --> C1["Retornar gbest, fitness, histórico"]
    
    style B1 fill:#533483,stroke:#e94560,color:#fff
    style B3 fill:#2b9348,stroke:#55a630,color:#fff
    style B5 fill:#16213e,stroke:#0f3460,color:#eee
    style B6 fill:#0f3460,stroke:#53a8b6,color:#fff
    style B10b fill:#d62828,stroke:#f77f00,color:#fff
    style B11 fill:#e76f51,stroke:#f4a261,color:#fff
    style B12b fill:#f4a261,stroke:#e76f51,color:#1a1a2e
    style C1 fill:#f4a261,stroke:#e76f51,color:#1a1a2e
```

---

## 5. Fluxo do QPSO Enhanced

```mermaid
flowchart TD
    subgraph INIT["Inicialização"]
        A1["Gerar partículas aleatórias\nparticles ∈ {0,1}^49"]
        A1b["Partícula 0 = referência\n(todas 49 boias ativas)"]
        A2["pbest = cópia inicial\ngbest = melhor global"]
        A3["Parâmetros Enhanced:\nlevy_mu = 0.06\ncrossover_rate = 0.35"]
    end
    
    A1 --> B1
    
    subgraph LOOP["Loop por max_iter iterações"]
        direction TB
        B1["📊 Avaliar fitness de\ntodas as partículas"]
        B1 --> B2["Atualizar pbest e gbest"]
        
        B2 --> B3["Calcular mbest\n(média das pbest, threshold 0.5)"]
        B3 --> B4["α = 1.0 − 0.5 × (t/T)\n(contração-expansão)"]
        
        B4 --> B4b["⭐ Salvar top-3 partículas\n(Elitismo)"]
        
        B4b --> B5{"⚛️ FASE 1: Multi-Estratégia\nSortear estratégia"}
        
        B5 -- "60% → Estratégia A\n(QPSO Global)" --> B6A["Atrator global:\np = φ·pbest + (1−φ)·gbest\nL = α·|mbest − x_i|"]
        
        B5 -- "40% → Estratégia B\n(Busca Local)" --> B6B["Atrator local:\np = φ·pbest + (1−φ)·particle\nL = α·0.5·|pbest − particle|"]
        
        B6A --> B7["Posição contínua:\nx = p ± L·ln(1/u)\n+ fallback se não-finito"]
        B6B --> B7
        
        B7 --> B8["Discretização:\nprobs = clip(sigmoid(x), 0.01, 0.99)\nbits = rand < probs ? 1 : 0"]
        
        B8 --> B9["🔀 FASE 2: Crossover com gbest\n(taxa decrescente)"]
        B9 --> B9b{"rand() <\np_cx?"}
        B9b -- Sim --> B9c["Criar filho:\nmáscara 50% do gbest\n+ 50% de pai aleatório"]
        B9c --> B9d{"Filho melhor\nque pior partícula?"}
        B9d -- Sim --> B9e["Substituir pior\npelo filho"]
        B9d -- Não --> B10
        B9e --> B10
        B9b -- Não --> B10
        
        B10["⭐ Reinserir Elite\n(top-3 restauradas)"]
        
        B10 --> B11["🦅 FASE 3: Lévy Flight\n(Pior metade da população)"]
        B11 --> B11b["Passo Lévy:\nstep = |u / |v|^0.1|\nFlip de n_flips bits"]
        
        B11b --> B12["🧬 FASE 4: Mutação Adaptativa\n(Híbrido GA)"]
        B12 --> B12b["Taxa adaptativa:\np_mut = 0.35 → 0.05"]
        
        B12b --> B13["🔧 FASE 5: Reparação Inteligente\n_repair_smart()"]
        B13 --> B13b["Ativar boias com MAIOR\nfrequência no pbest\n(não aleatória)"]
        
        B13b --> B14["📊 Re-verificação final\nAvaliar fitness de todas\nAtualizar gbest se melhorou"]
    end
    
    B14 --> C1["Retornar gbest, fitness, histórico"]
    
    style B5 fill:#533483,stroke:#e94560,color:#fff
    style B6A fill:#d62828,stroke:#f77f00,color:#fff
    style B6B fill:#0f3460,stroke:#53a8b6,color:#fff
    style B8 fill:#533483,stroke:#e94560,color:#fff
    style B9 fill:#e76f51,stroke:#f4a261,color:#fff
    style B4b fill:#f4a261,stroke:#e76f51,color:#1a1a2e
    style B10 fill:#f4a261,stroke:#e76f51,color:#1a1a2e
    style B11 fill:#d62828,stroke:#f77f00,color:#fff
    style B12 fill:#2b9348,stroke:#55a630,color:#fff
    style B13 fill:#16213e,stroke:#0f3460,color:#eee
    style B14 fill:#264653,stroke:#2a9d8f,color:#eee
    style C1 fill:#f4a261,stroke:#e76f51,color:#1a1a2e
```

---

## 6. Componente Híbrido: Mutação Adaptativa + Reparação Inteligente

```mermaid
flowchart LR
    A["População\natual"] --> A2["Calcular taxa:\np_mut = base − (base−final)×(t/T)\n(0.35→0.40 até 0.05)"]
    A2 --> B{"fitness_i\n< mediana?"}
    B -- Sim --> C{"rand()\n< p_mut?"}
    C -- Sim --> D["Inverter bit\naleatório"]
    C -- Não --> E["Manter\ninalterado"]
    B -- Não --> E
    D --> F{"Σ bits\n≥ min_boias?"}
    F -- Não --> G["🔧 Reparação Inteligente:\nAtivar boias com MAIOR\nfrequência no pbest"]
    G --> F
    F -- Sim --> H["População\nmutada + reparada"]
    E --> H
    
    style A fill:#1a1a2e,stroke:#e94560,color:#eee
    style A2 fill:#533483,stroke:#e94560,color:#fff
    style D fill:#e94560,stroke:#533483,color:#fff
    style G fill:#16213e,stroke:#0f3460,color:#eee
    style H fill:#2b9348,stroke:#55a630,color:#fff
```

---

## 7. Legenda de Cores

| Cor | Significado |
|-----|-------------|
| 🟣 Roxo | Componentes quânticos (rotação, colapso, superposição, multi-estratégia) |
| 🔴 Vermelho | Hamiltoniano de custo / dinâmica de Schrödinger / Lévy Flight |
| 🔵 Azul | Hamiltoniano mixer / atratores / reparação inteligente |
| 🟢 Verde | Componente genético híbrido (mutação GA adaptativa) |
| 🟠 Laranja | Elitismo / crossover / resultados e saídas |
| 🟤 Teal | Verificações finais / re-avaliação |

---

## 8. Mapeamento Código ↔ Fluxograma

| Bloco do Fluxograma | Arquivo | Linhas |
|:---|:---|:---:|
| Função de Fitness | `shared_utils.py` | 28–48 |
| Mutação Adaptativa | `shared_utils.py` | 51–79 |
| Taxa Adaptativa | `algoritmos_hibridos_enhanced.py` | 75–78 |
| Reparação Inteligente | `algoritmos_hibridos_enhanced.py` | 58–72 |
| Diversidade | `algoritmos_hibridos_enhanced.py` | 52–55 |
| QAOA Enhanced | `algoritmos_hibridos_enhanced.py` | 85–197 |
| QEA Enhanced | `algoritmos_hibridos_enhanced.py` | 204–315 |
| QPSO Enhanced | `algoritmos_hibridos_enhanced.py` | 322–475 |
| ML Surrogate | `shared_utils.py` | 104–155 |
