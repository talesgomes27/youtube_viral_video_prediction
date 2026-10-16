# Gate 2: Business Understanding, Hipóteses e Especificação Analítica (*Ask & Plan*)

- **Projeto:** `youtube_viral_video_prediction`
- **Arquétipo:** A — Data Science, Analytics & Machine Learning (`python-uv-quarto`)
- **Metodologia:** Modelo Híbrido CRISP-ML(Q) + Data Storytelling em 6 Seções
- **Etapa Atual:** Gate 2 — *Business Understanding, Hypotheses & Specs (Ask / Plan)*
- **Data:** 2026-10-15

---

## 1. Problema Central e Contexto de Negócio (*Business Understanding / Ask*)

O porte do canal inscrito ($\ln(1 + \text{subscriber\_count})$) explica isoladamente $65,5\%$ da variância logarítmica das visualizações no catálogo auditado ($r = +0,8097$, $N = 12.000$). Essa concentração lognormal distorce avaliações superficiais de desempenho editorial. Categorias de vídeos curtos como `Music` ($57,2\%$ de vídeos virais) e `Comedy` ($35,2\%$) parecem superar categorias longas por efeito de formato, quando na verdade reúnem canais com mediana de $2,06\text{ milhões}$ e $850\text{ mil}$ inscritos contra $198\text{ mil}$ em `Vlog`.

Estruturado sob a **Metodologia Híbrida CRISP-ML(Q) + Data Storytelling**, este projeto ataca três objetivos analíticos e de engenharia de *Machine Learning*:

1. **Arquitetura Preditiva Dual (`log1p(views)` + `viral` Calibrado):** Construir e validar modelos supervisionados capazes de prever tanto o volume contínuo $y_i = \ln(1 + \text{views}_i)$ (via regressão) quanto a probabilidade calibrada do evento `viral` ($\text{views}_i \ge 734.379$, top $20\%$), demonstrando a convergência entre classificadores nativos e regressores submetidos a corte no quantil $P_{80}$.
2. **Contraste de Regimes Temporais (*Cold Start* vs. *Early Signals*):** Quantificar o ganho real de generalização ao transitar do **Regime A (Pré-Publicação / Cold Start)** para o **Regime B (Pós-Publicação com Taxas Relativas Adimensionais)**, bloqueando por contrato as contagens absolutas (`likes`, `comments`, `shares`) que carregam *proxy leakage* ($r > 0,94$ com $\ln(1+\text{views})$).
3. **Modelagem de Desconfundimento e Explicabilidade (*Deconfounding & SHAP*):** Isolar o efeito líquido das variáveis controladas pelo criador (`thumbnail_face`, `tags_count`, `duration_sec`, `title_length`, `description_length`, `upload_hour`, `upload_day`) sobre o multiplicador relativo do canal $M_i = \ln(\text{views}_i / \text{subscriber\_count}_i)$, decompondo a importância por ganho (`gain`) e valores SHAP.

---

## 2. Questões Norteadoras e Hipóteses Estatísticas Formais

Formulamos cinco hipóteses falsificáveis que conectam a Seção 1 (*Business Understanding / Ask*) às seções de *EDA / Process* (Seção 3), *Modeling / Construct* (Seção 4) e *Evaluation & Explainability / Share* (Seção 5) do relatório Quarto:

- **Hipótese $H_1$ (Dominância Estrutural da Base Instalada):** O porte logarítmico do canal ($\ln(1 + \text{subscriber\_count})$) e a categoria editorial respondem por mais de $85\%$ da redução total de perda (*gain*) nos modelos de regressão de $\ln(1 + \text{views})$ e classificação de `viral`.
- **Hipótese $H_2$ (Paradoxo de Simpson na Duração do Vídeo):** A correlação marginal negativa entre `duration_sec` e $\ln(1 + \text{views})$ ($r = -0,2015$, $p < 10^{-4}$) é um artefato de confundimento induzido por `category` e `subscriber_count`. Ao controlar pelo porte do canal no alvo relativo $M_i = \ln(\text{views}_i / \text{subscriber\_count}_i)$, a correlação parcial de `duration_sec` anula-se ($|r| < 0,02$, $p > 0,05$).
- **Hipótese $H_3$ (Alavancas Acionáveis de Pré-Upload):** Controlado o tamanho do canal, a presença de rosto na miniatura (`thumbnail_face == 1`) e a densidade de tags (`tags_count`) elevam de forma estatisticamente significativa ($p < 0,001$ via Mann-Whitney U e regressão ajustada) tanto o multiplicador $\ln(\text{views}/\text{subscriber\_count})$ quanto a razão de chances (*odds ratio*) de viralidade.
- **Hipótese $H_4$ (Neutralidade do Agendamento Temporal):** A hora do upload (`upload_hour`, $0\text{--}23$) e o dia da semana (`upload_day`) não alteram significativamente a taxa de viralidade nem o multiplicador relativo do canal (testes Kruskal-Wallis e $\chi^2$ com $p > 0,05$ após correção para múltiplos testes).
- **Hipótese $H_5$ (Ortogonalidade das Taxas Pós-Upload no Regime B):** Eliminado o *proxy leakage* das contagens brutas, a adição das taxas relativas pós-upload (`like_rate`, `comment_rate`, `share_rate`) no **Regime B** produz ganho preditivo estatisticamente nulo ou residual ($\Delta \text{PR-AUC} < 0,005$ e $\Delta \text{RMSLE} < 0,005$) frente ao **Regime A (*Cold Start*)**.

---

## 3. Justificativa Matemática das Métricas e Esquema de Validação

### 3.1. Regressão Contínua (`log1p(views)` e Multiplicador Relativo)

Na escala bruta, `views` apresenta *skewness* de $+29,532$ e curtose excessiva de $1.305,18$, enquanto a transformação $y_i = \ln(1 + \text{views}_i)$ estabiliza a *skewness* em $+0,053$ e a curtose em $+0,080$. O uso de RMSE na escala bruta seria dominado por menos de $1\%$ de vídeos extremos ($> 10\text{M}$ de visualizações).

Definimos o **RMSLE (Root Mean Squared Logarithmic Error)** como métrica primária de regressão:

$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \ln(1 + \hat{v}_i) - \ln(1 + v_i) \right)^2} = \text{RMSE}(y, \hat{y})$$

Métricas secundárias obrigatórias na tabela multi-métrica:
1. **$\text{MAE}_{\ln}$:** Erro absoluto médio no espaço logarítmico.
2. **$R^2_{\ln}$:** Proporção de variância logarítmica explicada.
3. **$\text{MdAPE}$ (Median Absolute Percentage Error):** Calculado após a re-transformação não-enviesada de Duan ($\hat{v}_i = \exp(\hat{y}_i) \cdot \frac{1}{N}\sum \exp(e_j) - 1$) para tradução executiva direta.

### 3.2. Classificação Binária Calibrada (`viral`)

Com prevalência positiva fixa em $20,00\%$ ($2.400$ positivos e $9.600$ negativos), a acurácia simples é inadequada. Adotamos a **PR-AUC (Average Precision)** como métrica primária por concentrar a avaliação na recuperação e precisão da classe viral minoritária [@knaflic_2015].

Métricas secundárias obrigatórias:
1. **ROC-AUC:** Capacidade de discriminação global pareada.
2. **F1-Score no Quantil $P_{80}$ (`F1@P80`):** Avaliação no limiar operacional alinhado à priore populacional de $20\%$ ($\hat{p}_i \ge Q_{0,80}(\hat{p})$), complementado pelo `F1@0.50`.
3. **Brier Score:** Medida quadrática de calibração probabilística ($\frac{1}{N}\sum (\hat{p}_i - z_i)^2$).

### 3.3. Protocolo de Particionamento e Validação Cruzada

Por tratar-se de uma base *cross-sectional* sem *timestamp* calendário sequencial:
- **Holdout Estratificado Composto ($80\% / 20\%$):** Separação fixa (`SEED = 42`) de $9.600$ vídeos de treino e $2.400$ vídeos de teste estratificada pelos $30$ estratos conjuntos de `category` $\times$ `viral`.
- **5-Fold Stratified Cross-Validation:** Executada dentro do conjunto de treino ($9.600$ vídeos) com todos os transformadores (`StandardScaler`, `OneHotEncoder`) encapsulados dentro de `sklearn.pipeline.Pipeline` para impedir vazamento entre dobras.

---

## 4. Rastreabilidade 1:1 (*Business Understanding / Hipóteses* $\to$ *Evaluation & Storytelling / Share*)

Em conformidade com `/home/tales/.gemini/config/skills/pre-code-pipeline/references/quarto-quality-checklist.md`, a matriz abaixo vincula cada pergunta e hipótese da **Seção 1** às seções e figuras do relatório Quarto (`index.qmd`):

| ID | Questão Norteadora (*Business Understanding / Ask*) | Hipótese | Seção Quarto de Destino | Visualização Planejada (`1x2`, `set_box_aspect(1)`, `sns.despine()`, 300 DPI) |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | Como a distribuição lognormal de inscritos e visualizações se estrutura e onde incide o corte determinístico $P_{80}$? | $H_1$ | **Seção 3** (*Data Prep & EDA / Process*) | **Figura 1 (`fig01_distributions_and_threshold.png`):** `(a)` Histograma/KDE de $\log_{10}(\text{views})$ com fronteira $P_{80}$ ($734.379$ views); `(b)` Dispersão densa $\log_{10}(\text{subs})$ vs. $\log_{10}(\text{views})$ ($r = +0,81$). |
| **Q2** | Por que `Music` e `Comedy` lideram a taxa bruta de viralidade e vídeos curtos realmente geram mais alcance relativo? | $H_2$ | **Seção 3** (*Data Prep & EDA / Process*) | **Figura 2 (`fig02_simpson_paradox_categories.png`):** `(a)` Taxa de viralidade (%) vs. Mediana de inscritos por categoria; `(b)` Decomposição do Paradoxo de Simpson: `duration_sec` vs. $\ln(\text{views})$ ($r = -0,20$) e vs. $\ln(\text{views}/\text{subs})$ ($r = -0,01$). |
| **Q3** | Quais alavancas pré-upload (`thumbnail_face`, `tags_count`, horário e dia) geram ganho líquido descontado o porte do canal? | $H_3, H_4$ | **Seção 3** (*Data Prep & EDA / Process*) | **Figura 3 (`fig03_creator_levers_and_timing.png`):** `(a)` Efeito de `thumbnail_face` e `tags_count` sobre $\ln(\text{views}/\text{subs})$ por quartil de inscritos; `(b)` Perfil horário (`upload_hour` $\times$ `upload_day`) comprovando a neutralidade temporal. |
| **Q4** | Qual família de algoritmos vence na predição dual e como o regressor limiarizado em $P_{80}$ se compara ao classificador direto? | $H_1, H_3$ | **Seção 4** (*Modeling / Construct*) & **Seção 5** (*Evaluation / Share*) | **Figura 4 (`fig04_dual_modeling_performance.png`):** `(a)` Curvas Precision-Recall (PR-AUC) no teste ($N=2.400$) incluindo o regressor dual; `(b)` Calibração contínua $\log_{10}(\text{views})$ predito vs. real com quadrantes de decisão em $P_{80}$. |
| **Q5** | As taxas relativas pós-upload (**Regime B**) superam o modelo de *Cold Start* (**Regime A**) e quais variáveis dominam a explicabilidade (`gain` / SHAP)? | $H_5$ | **Seção 5** (*Evaluation, Diagnostics & Explainability / Share*) | **Figura 5 (`fig05_scenario_a_vs_b_importance.png`):** `(a)` Comparativo multi-métrica (5-Fold CV + Teste) entre Regime A e Regime B; `(b)` Importância de variáveis por `gain` / SHAP confirmando a hierarquia de preditores. |

---

## 5. Requisitos Funcionais (RF), Não-Funcionais (RNF) e *Non-Goals*

### 5.1. Requisitos Funcionais e Não-Funcionais

- **RF-01 (Contratos de Dados):** Validar `DC-01` a `DC-07` em `src/youtube_viral/contracts.py` antes do estágio *Process*.
- **RF-02 (Isolamento de Regimes A e B):** Gerar matrizes de features estritamente isoladas para o Regime A (*Cold Start*) e Regime B (*Early Signals Relativos*) em `src/youtube_viral/features.py`.
- **RF-03 (Modelagem Tripla e Explicabilidade):** Treinar Baselines Lineares (`LogisticRegression` / `Ridge`), `RandomForest` e `LightGBM` para Classificação (`viral`), Regressão (`log1p(views)`) e Desconfundimento (`log(views/subs)`), extraindo importância por `gain` e valores SHAP em `src/youtube_viral/modeling.py`.
- **RF-04 (Relatório Quarto nas 6 Seções CRISP-ML(Q) + Storytelling):** Estruturar `index.qmd` exatamente nas 6 seções canônicas do `quarto-quality-checklist.md`.
- **RNF-01 (Ambiente e Qualidade):** Execução via `/home/tales/.local/bin/uv`, linting via `ruff`, testes via `pytest` e compilação via `/opt/quarto/bin/quarto` com `lang: pt-BR` e `execute: freeze: auto`.
- **RNF-02 (Redação Sênior Anti-IA):** Conformidade integral com `writing-pipeline` e [@zinsser_1976].

### 5.2. Escopo Negativo Explícito (*Non-Goals*)

1. **Coleta Externa na API do YouTube:** Restrito exclusivamente aos $12.000$ registros de `viral_videos.csv`.
2. **NLP ou Visão Computacional Bruta:** Não há títulos textuais brutos ou arquivos de imagem de *thumbnails* na base; a modelagem opera sobre as features tabulares fornecidas.
3. **Modelagem de Séries Temporais:** Inaplicável dada a natureza transversal (*cross-sectional*) do dataset.
4. **Serviços Web, Bancos de Dados ou Dashboards SaaS (Arquétipo B):** O entregável final é o pipeline científico testado e o relatório analítico em Quarto HTML.
