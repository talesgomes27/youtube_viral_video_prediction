# Predição de Vídeos Virais no YouTube: Modelagem Dual, Paradoxo de Simpson e Desconfundimento de Audiência

Pipeline analítico e preditivo construído sobre $12.000$ vídeos distribuídos em $15$ categorias editoriais (`viral_videos.csv`). O projeto investiga os determinantes de alcance orgânico ($\ln(1 + \text{views})$) e de viralidade (`viral = 1`, definido pelo corte determinístico no percentil $80$, $\text{views} \ge 734.379$), isolando o efeito inercial do tamanho do canal (`subscriber_count`) das alavancas acionáveis pelo criador antes da publicação.

A arquitetura segue o **Arquétipo A (`python-uv-quarto`)** e a **Metodologia Híbrida CRISP-ML(Q) + Data Storytelling em 6 Seções**, combinando contratos de validação em tempo de execução (`DC-01` a `DC-07`), bloqueio estrito de *target* e *proxy leakage*, validação cruzada estratificada (5-Fold CV + Holdout 80/20) e explicabilidade via ganho de informação (`importance_type="gain"`) e `TreeSHAP`.

---

## Principais Resultados Auditados

1. **Dominância da Base de Inscritos ($H_1$):** $\ln(1 + \text{subscriber\_count})$ possui correlação linear de $r = 0,8097$ ($p < 10^{-300}$) com $\ln(1 + \text{views})$, explicando isoladamente $65,56\%$ da variância ($R^2 = 0,6556$). No `LightGBM` de pré-publicação (Regime A), `subscriber_count` e `log_subscriber_count` concentram **$93,59\%$** do ganho total na regressão e **$80,09\%$** na classificação.
2. **Paradoxo de Simpson na Duração do Vídeo ($H_2$):** A correlação marginal negativa entre `duration_sec` e $\ln(1 + \text{views})$ ($r = -0,2015, p = 3,97 \times 10^{-110}$) decorre do confundimento por categoria: nichos de vídeos curtos (`Music`: mediana $239\text{s}$, $57,2\%$ viral; `Comedy`: $297\text{s}$, $35,2\%$ viral) possuem canais $5\text{x}$ a $10\text{x}$ maiores que categorias mais longas (`Travel`, `Vlog`). Ao normalizar pelo porte do canal ($\ln(\text{views}/\text{subscriber\_count})$), a correlação cai para **$r = -0,0077$ ($p = 0,3966$)**.
3. **Alavancas Acionáveis do Criador ($H_3$):** A presença de rosto na miniatura (`thumbnail_face = 1`) eleva a taxa de viralidade de **$18,15\%$ para $22,78\%$** ($+25,52\%$ relativo, $p = 9,34 \times 10^{-32}$) e aumenta a razão de visualizações por inscrito em **$+24,39\%$**. O volume de tags (`tags_count`) mantém sinal positivo significativo após o desconfundimento ($r = +0,0549, p = 1,83 \times 10^{-9}$).
4. **Neutralidade do Agendamento Temporal ($H_4$):** Horário de upload (`upload_hour`, Kruskal-Wallis $p = 0,6606$) e dia da semana (`upload_day`, Kruskal-Wallis $p = 0,8717$) não afetam o alcance nem a probabilidade de viralização.
5. **Previsão Cold Start vs. Pós-Upload Relativo ($H_5$) e Convergência Dual ($P_{80}$):** Contagens absolutas pós-upload (`likes`, `comments`, `shares`) são bloqueadas pelo contrato `DC-07` por apresentarem *proxy leakage* severo ($r \in [0,9495;\, 0,9736]$ com $\ln(1+\text{views})$). Quando utilizamos apenas taxas relativas de engajamento (Regime B), o ganho sobre o modelo de pré-publicação (Regime A) é residual ($\Delta \text{PR-AUC} = +0,0051$ e $\Delta \text{RMSLE} = -0,0006, p = 0,8331$).

### Síntese de Performance no Conjunto de Teste Holdout ($N = 2.400$)

| Tarefa | Regime | Modelo | CV Primária (Média ± DP) | Teste PR-AUC | Teste ROC-AUC | Teste F1@P80 | Teste Brier | Teste RMSLE | Teste $R^2_{\ln}$ | Teste MdAPE (%) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Classificação** | **A (Cold Start)** | `Baseline (Categoria)` | 0,3666 ± 0,0006 | 0,3671 | 0,6906 | 0,4090 | 0,1453 | - | - | - |
| **Classificação** | **A (Cold Start)** | `LogisticRegression` | **0,7357 ± 0,0176** | **0,7407** | **0,9086** | **0,6701** | **0,0910** | - | - | - |
| **Classificação** | **A (Cold Start)** | `RandomForest` | 0,7202 ± 0,0189 | 0,7265 | 0,9056 | 0,6660 | 0,0924 | - | - | - |
| **Classificação** | **A (Cold Start)** | `LightGBM` | 0,7234 ± 0,0190 | 0,7185 | 0,9048 | 0,6493 | 0,0937 | - | - | - |
| **Classificação** | **A (Cold Start)** | `LightGBM_DualRegressor` | 0,7329 ± 0,0171 | 0,7383 | 0,9072 | 0,6639 | 0,0929 | - | - | - |
| **Classificação** | **B (Relativo)** | `LogisticRegression` | 0,7351 ± 0,0180 | 0,7402 | 0,9087 | **0,6701** | **0,0911** | - | - | - |
| **Classificação** | **B (Relativo)** | `LightGBM` | 0,7272 ± 0,0192 | 0,7237 | 0,9060 | 0,6576 | 0,0935 | - | - | - |
| **Classificação** | **B (Relativo)** | `LightGBM_DualRegressor` | 0,7345 ± 0,0175 | **0,7409** | **0,9094** | 0,6576 | 0,0922 | - | - | - |
| **Regressão** | **A (Cold Start)** | `Baseline (Categoria)` | 1,5242 ± 0,0104 | - | - | - | - | 1,5269 | 0,1438 | 204,64% |
| **Regressão** | **A (Cold Start)** | `Ridge` | **0,9592 ± 0,0121** | - | - | - | - | **0,9457** | **0,6716** | 69,96% |
| **Regressão** | **A (Cold Start)** | `RandomForest` | 0,9749 ± 0,0122 | - | - | - | - | 0,9569 | 0,6637 | **69,90%** |
| **Regressão** | **A (Cold Start)** | `LightGBM` | 0,9697 ± 0,0108 | - | - | - | - | 0,9584 | 0,6627 | 70,04% |
| **Regressão** | **B (Relativo)** | `Ridge` | **0,9595 ± 0,0118** | - | - | - | - | **0,9454** | **0,6718** | 69,91% |
| **Regressão** | **B (Relativo)** | `LightGBM` | 0,9693 ± 0,0089 | - | - | - | - | 0,9578 | 0,6631 | **69,21%** |

---

## Arquitetura do Repositório

```text
youtube_viral_video_prediction/
├── _quarto.yml                                  # Configuração Quarto (lang: pt-BR, freeze: auto, cosmo + styles.scss)
├── styles.scss                                  # Folha de estilo customizada para o relatório executivo
├── index.qmd                                    # Relatório Quarto nas 6 Seções Canônicas do CRISP-ML(Q) + Storytelling
├── pyproject.toml                               # Manifesto PEP 621 gerenciado via uv (hatchling, ruff, pytest)
├── AGENTS.md                                    # Diretrizes operacionais e guardrails anti-leakage
├── README.md                                    # Documentação executiva e operacional do projeto
├── datasheet.md                                 # Documentação de origem da base sintética
├── viral_videos.csv                             # Dataset bruto imutável (12.000 linhas x 15 colunas)
├── task.md                                      # Rastreamento de Gates (pre-code-pipeline e execution-pipeline)
├── docs/
│   ├── 00_archetype.md                          # Declaração formal do Arquétipo A
│   ├── 01-data-discovery-and-contracts.md       # Gate 1: Data Understanding, diagnóstico de leakage e contratos DC-01..DC-07
│   ├── 02-PRD-and-specs.md                      # Gate 2: Business Understanding, hipóteses H1-H5 e matriz de rastreabilidade
│   ├── 03-architecture-and-schema.md            # Gate 3: Arquitetura do pipeline e storyboard de figuras 300 DPI
│   ├── 04-scaffolding-and-conventions.md        # Gate 4: Estrutura física e convenções técnicas
│   ├── 05-implementation-plan.md                # Gate 5: Grafo de precedência TDD e plano de execução
│   └── adr/
│       ├── 0001-dual-target-and-anti-leakage-architecture.md
│       └── 0002-stack-uv-sklearn-lightgbm-quarto.md
├── src/
│   └── youtube_viral/
│       ├── __init__.py                          # Semente determinística global (SEED = 42)
│       ├── contracts.py                         # Contratos DC-01 a DC-07 e bloqueio anti-leakage por regime
│       ├── features.py                          # Split estratificado 80/20 e construção dos Regimes A e B
│       ├── evaluation.py                        # Métricas (RMSLE, Duan smearing, PR-AUC, F1@P80) e testes H1-H5
│       ├── modeling.py                          # Treinamento 5-Fold CV, Convergência Dual P80 e Gain/TreeSHAP
│       ├── plotting.py                          # Geração das 5 figuras de publicação (painéis 1x2, 300 DPI)
│       └── pipeline.py                          # Orquestrador end-to-end para exportação em reports/
├── tests/
│   ├── conftest.py                              # Fixtures determinísticas compartilhadas
│   ├── test_contracts.py                        # Testes dos contratos DC-01..DC-07 e barreiras de leakage
│   ├── test_features.py                         # Testes de engenharia de atributos e partição 80/20
│   ├── test_evaluation.py                       # Testes de métricas e validação das hipóteses H1-H5
│   ├── test_modeling.py                         # Testes de modelagem supervisionada, Dual P80 e Gain/SHAP
│   └── test_pipeline_outputs.py                 # Testes de integridade de JSON/CSV e resolução das figuras PNG
└── reports/
    ├── figures/                                 # 5 figuras PNG (3600x1560 px, 300 DPI)
    └── metrics/                                 # summary_results.json e model_comparison.csv
```

---

## Como Reproduzir o Projeto

### 1. Provisionar o Ambiente Virtual (`uv`)

```bash
/home/tales/.local/bin/uv sync
```

### 2. Executar o Pipeline Completo (Contratos, Hipóteses, Modelos e Figuras 300 DPI)

```bash
/home/tales/.local/bin/uv run python -m youtube_viral.pipeline
```

### 3. Executar a Suíte de Testes (`pytest`) e Qualidade de Código (`ruff`)

```bash
/home/tales/.local/bin/uv run pytest
/home/tales/.local/bin/uv run ruff check .
/home/tales/.local/bin/uv run ruff format --check .
```

### 4. Renderizar o Relatório Executivo HTML (`Quarto`)

```bash
/opt/quarto/bin/quarto render index.qmd
```

O relatório interativo compilado será gravado em `_site/index.html`.

---

## Licença

Distribuído sob os termos da licença MIT. Veja [LICENSE](LICENSE).
