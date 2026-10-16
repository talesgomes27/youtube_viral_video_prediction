# ADR-0002: Padronização da Stack Científica (`uv`, Scikit-Learn, LightGBM e Quarto Desacoplado)

- **Status:** Aceito
- **Data:** 2026-10-15
- **Projeto:** `youtube_viral_video_prediction` (Arquétipo A)

---

## Contexto

O projeto exige reprodutibilidade determinística, execução rápida em ambiente local Linux sem necessidade de GPU ou compilação C/C++ complexa, validação automatizada por testes (`pytest`), linting estrito (`ruff`) e geração de um relatório executivo em português (`lang: pt-BR`) via Quarto CLI (`/opt/quarto/bin/quarto`).

Se o treinamento de modelos e a geração de gráficos pesados ocorrerem apenas dentro de células embutidas durante o `quarto render`, torna-se difícil testar unitariamente cada etapa do pipeline via `pytest` e auditar os números exatos antes de redigir a narrativa.

---

## Decisão

1. **Gerenciamento de Runtimes e Pacotes:** Utilizar exclusivamente `uv` (`/home/tales/.local/bin/uv`) com `pyproject.toml` (PEP 621), sem uso de Conda/Miniforge, pois todas as bibliotecas necessárias possuem *wheels* binárias puras e estáveis no PyPI.
2. **Bibliotecas de Modelagem e Estatística:** Padronizar em `pandas`, `numpy`, `scipy` (testes paramétricos e não-paramétricos), `scikit-learn` (pipelines, pré-processamento, validação cruzada, modelos lineares e Random Forest) e `lightgbm` (Gradient Boosting com extração de importância por `gain`).
3. **Arquitetura de Renderização Desacoplada (Artifact-First Quarto):**
   - Os módulos em `src/youtube_viral/` executam o pipeline completo, salvam as tabelas de métricas auditadas em `reports/metrics/` (JSON/CSV) e exportam as 5 figuras pareadas em `reports/figures/*.png` (300 DPI, `set_box_aspect(1)`, `sns.despine()`).
   - O documento Quarto (`index.qmd`) consome e exibe esses artefatos validados com `execute: freeze: auto` ativo no `_quarto.yml`.

---

## Consequências

### Positivas
- **Testabilidade Total (`pytest`):** Cada função de contrato de dados, transformação, teste estatístico e treinamento é validada por testes automatizados independentes do Quarto.
- **Consistência entre Prosa e Gráficos:** Como as métricas e importâncias (`importance_type='gain'`) são persistidas em JSON/CSV antes da escrita final de `index.qmd`, elimina-se o risco de divergência entre os números citados no texto e os valores renderizados nas figuras.
- **Builds Rápidos:** `quarto render` compila o relatório HTML em poucos segundos sem reexecutar treinamentos desnecessários.

### Negativas / Trade-offs
- Requer um script orquestrador (`src/youtube_viral/pipeline.py`) para sincronizar a geração dos artefatos em `reports/` antes de invocar `quarto render`.
