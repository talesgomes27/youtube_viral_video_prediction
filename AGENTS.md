# AGENTS.md — Diretrizes Operacionais e Guardrails do Repositório

- **Projeto:** `youtube_viral_video_prediction`
- **Arquétipo:** `A` — Data Science, Analytics & Machine Learning (`python-uv-quarto`)
- **Metodologia:** Modelo Híbrido CRISP-ML(Q) + Data Storytelling em 6 Seções
- **Contrato de Herança:** `docs/00_archetype.md`

---

## 1. Ambiente, Binários e Execução Isolada

1. **Gerenciador de Pacotes e Python (`uv`):**
   - Caminho canônico: `/home/tales/.local/bin/uv`
   - **Proibido** executar `pip install`, `conda` ou ativar `.venv` manualmente com `source .venv/bin/activate`.
   - Sempre execute ferramentas dentro do ambiente gerenciado:
     - Testes: `/home/tales/.local/bin/uv run pytest`
     - Linter e Formatador: `/home/tales/.local/bin/uv run ruff check . --fix && /home/tales/.local/bin/uv run ruff format .`
     - Pipeline Analítico: `/home/tales/.local/bin/uv run python -m youtube_viral.pipeline`
2. **Quarto CLI e Git:**
   - Quarto CLI: `/opt/quarto/bin/quarto` (renderização via `/opt/quarto/bin/quarto render index.qmd`)
   - Git: `/usr/bin/git`

---

## 2. Guardrails Inegociáveis de Ciência de Dados (Anti-Leakage)

1. **Fronteira Determinística ($P_{80}$):** A coluna `viral` é definida por $\text{views} \ge 734.379$. A variável `views` (ou qualquer transformação monotônica como `log_views` ou `views_per_sub`) **JAMAIS** pode entrar na matriz de atributos $X$ ao prever `viral`.
2. **Proibição de Contagens Brutas de Engajamento (*Proxy Leakage*):** As colunas `likes`, `comments` e `shares` em valores absolutos são geradas multiplicando `views` por taxas limitadas ($r > 0,94$ em escala `log1p`). Seu uso bruto como feature preditiva é estritamente proibido em qualquer regime (`DC-07`).
3. **Isolamento de Regimes Temporais:**
   - **Regime A (Cold Start / Pré-Upload):** Restrito exclusivamente a metadados pré-publicação e `subscriber_count`.
   - **Regime B (Early Signals / Pós-Upload):** Adiciona apenas razões adimensionais (`like_rate`, `comment_rate`, `share_rate`, `engagement_rate_total`, `comments_per_like`, `shares_per_like`).
4. **Determinismo Global:** Todas as divisões de dados (`StratifiedKFold`, `train_test_split`), algoritmos de árvore (`RandomForest`, `LightGBM`) e reamostragens de bootstrap devem fixar `random_state = 42`.

---

## 3. Padrão Editorial e Visual (CRISP-ML(Q) + Storytelling em Quarto)

1. **Estrutura Canônica de 6 Seções em `index.qmd` (`quarto-quality-checklist.md`):**
   - `## Contexto de Negócio e Hipóteses` (*Business Understanding / Ask*)
   - `## Auditoria de Dados e Contratos de Qualidade` (*Data Understanding / Prepare*)
   - `## Análise Exploratória e Engenharia de Atributos` (*Data Prep & EDA / Process*)
   - `## Modelagem Estatística e Machine Learning` (*Modeling / Construct*)
   - `## Avaliação, Diagnóstico e Explicabilidade` (*Evaluation / Share*)
   - `## Recomendações Executivas e Plano de Ação` (*Deployment / Act*)
2. **Regras Gráficas e Técnicas:**
   - Idioma: `lang: pt-BR` e `execute: freeze: auto` no `_quarto.yml`.
   - **Zero numeração manual** nos títulos Markdown de `index.qmd` (`number-sections: true` já numera automaticamente).
   - Painéis de subfiguras lado a lado (`1x2`) devem usar `figsize=(12, 5.2)`, `constrained_layout=True`, `ax.set_box_aspect(1)` em ambos os eixos, `sns.despine()` e `dpi=300`.
   - Importância de atributos em árvores (`LightGBM`) deve usar `importance_type='gain'` (acompanhada de contribuições TreeSHAP) e coincidir exatamente com a ordem citada na narrativa.
3. **Redação Técnica (`writing-pipeline` & `humanizer`):**
   - Abrir seções e parágrafos com o resultado numérico ou fato empírico concreto.
   - Alternar comprimento de frases (*burstiness*) e eliminar clichês promocionais (*"robusto"*, *"vital"*, *"crucial"*, *"não apenas X, mas também Y"* e travessões duplos `—` conectando orações).
   - Mensagens de commit semânticas, em minúsculas e imperativas (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
