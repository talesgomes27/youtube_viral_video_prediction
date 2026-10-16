# Changelog

Todas as mudanças relevantes deste projeto são documentadas neste arquivo.

O formato segue o padrão [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e este projeto adota o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [0.1.0] - 2026-10-15

### Adicionado

- **Governança e Especificações (`docs/`, `AGENTS.md`, `pyproject.toml`):**
  - Arquitetura de planejamento pré-código em 5 Gates (`00_archetype.md` a `05-implementation-plan.md`) e registros de decisão arquitetural (`ADR-0001` e `ADR-0002`) sob a metodologia híbrida CRISP-ML(Q) + Data Storytelling em 6 seções.
  - Configuração PEP 621 gerenciada via `uv` com `pytest`, `ruff`, `scikit-learn`, `lightgbm`, `scipy`, `matplotlib` e `seaborn`.
- **Contratos de Dados e Blindagem Anti-Leakage (`src/youtube_viral/contracts.py`):**
  - Validação em tempo de execução dos contratos `DC-01` a `DC-07` sobre `viral_videos.csv` ($12.000$ vídeos, $15$ categorias, zero valores nulos e fronteira determinística $P_{80}$ em $734.379$ visualizações).
  - Bloqueio programático contra *target leakage* (`views` $\leftrightarrow$ `viral`) e *proxy leakage* de contagens brutas pós-upload (`likes`, `comments`, `shares`, que exibem $r \in [0,9495;\, 0,9736]$ com $\ln(1 + \text{views})$).
- **Engenharia de Atributos e Isolamento de Regimes (`src/youtube_viral/features.py`):**
  - Particionamento determinístico 80/20 (`SEED = 42`, $9.600$ treino / $2.400$ teste) estratificado pela chave composta `category + "_" + viral`.
  - Construção separada do **Regime A (*Cold Start / Pré-Publicação*, 18 atributos)** e do **Regime B (*Sinais Relativos Pós-Upload*, 24 atributos)**.
- **Avaliação Estatística e Modelagem Supervisionada Tripla (`src/youtube_viral/evaluation.py`, `src/youtube_viral/modeling.py`):**
  - Testes formais das hipóteses $H_1$ (dominância de inscritos, $R^2 = 0,6556$), $H_2$ (decomposição do Paradoxo de Simpson em `duration_sec`, onde $r = -0,2015$ colapsa para $r = -0,0077, p = 0,3966$), $H_3$ (ganho relativo de $+25,52\%$ em `thumbnail_face` e correlação positiva de `tags_count`), $H_4$ (neutralidade de hora e dia de upload) e $H_5$ (equivalência prática entre Regime A e Regime B).
  - Estimação em 5-Fold CV estratificado e Teste Holdout para regressão logarítmica (`Baseline`, `Ridge`, `RandomForest`, `LightGBM` com correção não-paramétrica de Duan), classificação calibrada (`LogisticRegression`, `RandomForest`, `LightGBM`), **Convergência Dual no quantil $P_{80}$** (`LightGBM_DualRegressor`, `PR-AUC = 0,7383` no Regime A e `0,7409` no Regime B) e modelo desconfundido de alcance por inscrito ($\ln(\text{views}/\text{subscriber\_count})$).
  - Extração de importância por ganho de informação (`importance_type="gain"`) e valores exatos de contribuição marginal `TreeSHAP`.
- **Visualização Científica 300 DPI e Relatório Quarto (`src/youtube_viral/plotting.py`, `index.qmd`, `_quarto.yml`):**
  - Geração automatizada de 5 painéis duplos `1x2` nivelados (`3600 x 1560 px`, 300 DPI) em `reports/figures/` e consolidação de métricas em `reports/metrics/`.
  - Relatório executivo `index.qmd` estruturado nas 6 seções canônicas do CRISP-ML(Q) + Data Storytelling, com cache congelado em `_freeze/` e suíte de 32 testes automatizados (`pytest`).
