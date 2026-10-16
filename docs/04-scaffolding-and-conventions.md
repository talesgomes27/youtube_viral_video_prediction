# Gate 4: Scaffolding, Estrutura de Pastas e Convenções

- **Projeto:** `youtube_viral_video_prediction`
- **Arquétipo:** A — Data Science, Analytics & Machine Learning (`python-uv-quarto`)
- **Metodologia:** Modelo Híbrido CRISP-ML(Q) + Data Storytelling em 6 Seções
- **Data:** 2026-10-15

---

## 1. Árvore Física do Repositório

A estrutura de diretórios separa especificações arquiteturais (`docs/`), código-fonte modular (`src/youtube_viral/`), suíte de testes unitários e contratuais (`tests/`), artefatos estáticos auditados (`reports/`) e a camada de publicação executiva (`index.qmd` e `_quarto.yml`):

```text
youtube_viral_video_prediction/
├── _quarto.yml                                  # Configuração Quarto (lang: pt-BR, freeze: auto, cosmo + styles.scss)
├── styles.scss                                  # Folha de estilo customizada para o relatório executivo
├── index.qmd                                    # Relatório Quarto estruturado nas 6 Seções Canônicas do CRISP-ML(Q) + Storytelling
├── pyproject.toml                               # Manifesto PEP 621 gerenciado via uv (hatchling, ruff, pytest)
├── AGENTS.md                                    # Regras operacionais, caminhos canônicos e guardrails anti-leakage
├── README.md                                    # Documentação do repositório
├── datasheet.md                                 # Datasheet original da base sintética
├── viral_videos.csv                             # Dataset bruto imutável (12.000 linhas x 15 colunas)
├── task.md                                      # Memória de trabalho e rastreamento dos Gates
├── docs/
│   ├── 00_archetype.md                          # Contrato de herança (Arquétipo A - CRISP-ML(Q) + Storytelling)
│   ├── 01-data-discovery-and-contracts.md       # Gate 1: Data Understanding, diagnóstico de leakage e contratos DC-01..DC-07
│   ├── 02-PRD-and-specs.md                      # Gate 2: Business Understanding, hipóteses H1-H5, métricas e matriz 1:1
│   ├── 03-architecture-and-schema.md            # Gate 3: Topologia Mermaid, Regimes A/B, modelos e storyboard de figuras
│   ├── 04-scaffolding-and-conventions.md        # Gate 4: Estrutura física e convenções (este documento)
│   ├── 05-implementation-plan.md                # Gate 5: Grafo de tarefas atômicas e plano TDD (Deployment / Act)
│   └── adr/
│       ├── 0001-dual-target-and-anti-leakage-architecture.md
│       └── 0002-stack-uv-sklearn-lightgbm-quarto.md
├── src/
│   └── youtube_viral/
│       ├── __init__.py                          # Constante global SEED = 42 e versão
│       ├── contracts.py                         # (Fase Execução) Validação DC-01..DC-07 e guarda anti-leakage
│       ├── features.py                          # (Fase Execução) Split estratificado e matrizes Regime A / B
│       ├── evaluation.py                        # (Fase Execução) Testes estatísticos H1..H5 e métricas
│       ├── modeling.py                          # (Fase Execução) Treinamento CV 5-Fold, Dual P80 e Gain/TreeSHAP
│       ├── plotting.py                          # (Fase Execução) Geração das 5 figuras 1x2 em 300 DPI
│       └── pipeline.py                          # (Fase Execução) Orquestrador CLI para gerar reports/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                              # (Fase Execução) Fixtures compartilhadas
│   ├── test_contracts.py                        # (Fase Execução) Testes de schema e bloqueio de leakage
│   ├── test_features.py                         # (Fase Execução) Testes de transformações e isolamento Regime A/B
│   ├── test_evaluation.py                       # (Fase Execução) Testes de métricas e hipóteses H1-H5
│   ├── test_modeling.py                         # (Fase Execução) Testes de reprodutibilidade, Dual P80 e Gain/SHAP
│   └── test_pipeline_outputs.py                 # (Fase Execução) Testes de integridade de figuras e JSONs
└── reports/
    ├── figures/                                 # Destino das 5 figuras PNG (300 DPI) consumidas pelo Quarto
    └── metrics/                                 # Destino dos JSON/CSV com resultados estatísticos e de modelos
```

---

## 2. Convenções de Código, Qualidade e Quarto

1. **Padronização Python (`pyproject.toml`):**
   - Compatibilidade Python `>=3.11`, empacotamento `src-layout` via `hatchling`.
   - `ruff` configurado com comprimento de linha de `100` caracteres e regras `["E", "F", "I", "N", "W", "UP", "B", "SIM"]` (permitindo `X` e `X_train` matemáticos via `ignore = ["N803", "N806"]`).
   - `pytest` configurado com `pythonpath = ["src"]` e descoberta automática em `tests/`.
2. **Padronização Quarto (`_quarto.yml` e `index.qmd`):**
   - `lang: pt-BR` ativo para renderizar automaticamente `Autor`, `Data`, `Sumário Executivo` e `Figura` em português.
   - `number-sections: true` ativo; portanto, nenhum cabeçalho Markdown em `index.qmd` contém numeração manual.
   - Estruturado nas **6 Seções Canônicas do Modelo Híbrido CRISP-ML(Q) + Data Storytelling** (`Contexto de Negócio e Hipóteses`, `Auditoria de Dados e Contratos de Qualidade`, `Análise Exploratória e Engenharia de Atributos`, `Modelagem Estatística e Machine Learning`, `Avaliação, Diagnóstico e Explicabilidade`, `Recomendações Executivas e Plano de Ação`).
   - `execute: freeze: auto` e `fig-dpi: 300` configurados para reprodutibilidade e nitidez gráfica.
