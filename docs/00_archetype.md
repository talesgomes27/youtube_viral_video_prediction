---
archetype: A
stack: python-uv-quarto
project_name: youtube_viral_video_prediction
gerado_por: pre-code-pipeline
data: 2026-10-15
---

# Contrato de Herança Arquitetural (`00_archetype.md`)

Este documento congela o arquétipo operacional e a stack tecnológica do projeto **`youtube_viral_video_prediction`**, orientando o roteamento polimórfico das meta-skills (`pre-code-pipeline`, `execution-pipeline`, `writing-pipeline` e `release-and-ops-pipeline`).

## Diretrizes de Governança Aplicáveis

- **Arquétipo:** `A` — Data Science, Analytics & Machine Learning (Python / Quarto).
- **Metodologia Analítica:** **Modelo Híbrido CRISP-ML(Q) + Data Storytelling em 6 Seções**:
  1. *Business Understanding & Hypotheses (Ask)*
  2. *Data Understanding & Contracts (Prepare)*
  3. *Data Prep, EDA & Feature Engineering (Process)*
  4. *Statistical & ML Modeling (Construct)*
  5. *Evaluation, Diagnostics & Explainability (Share)*
  6. *Deployment & Executive Action (Act)*
- **Gerenciamento de Ambiente e Qualidade:** `uv` (`/home/tales/.local/bin/uv`) + `pyproject.toml` (PEP 621) + `Ruff` + `pytest`.
- **Validação de Dados:** Contratos de dados estritos em tempo de execução (`DC-01` a `DC-07`) antes do estágio *Data Prep & Feature Engineering (Process)*.
- **Publicação Científica e Executiva:** Quarto CLI (`/opt/quarto/bin/quarto`), `lang: pt-BR`, `execute: freeze: auto`, conformidade total com `/home/tales/.gemini/config/skills/pre-code-pipeline/references/quarto-quality-checklist.md`.
