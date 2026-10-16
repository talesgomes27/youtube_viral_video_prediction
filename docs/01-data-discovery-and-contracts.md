# Gate 1: Data Understanding, Diagnóstico de Leakage e Contratos de Dados (*Prepare*)

- **Projeto:** `youtube_viral_video_prediction`
- **Arquétipo:** A — Data Science, Analytics & Machine Learning (`python-uv-quarto`)
- **Metodologia:** Modelo Híbrido CRISP-ML(Q) + Data Storytelling em 6 Seções
- **Etapa Atual:** Gate 1 — *Data Understanding & Contracts (Prepare)* (com síntese preliminar de *Business Understanding / Ask*)
- **Data da Auditoria:** 2026-10-15

---

## 1. Síntese Preliminar de Negócio (*Business Understanding / Ask*)

A base `viral_videos.csv` modela o desempenho de $12.000$ vídeos em $15$ categorias de conteúdo no formato do YouTube, relacionando metadados de pré-publicação, porte do canal (`subscriber_count`), métricas de engajamento da audiência (`likes`, `comments`, `shares`) e dois desfechos de sucesso: o volume total de visualizações (`views`, contínuo) e o status de vídeo viral (`viral`, binário, definido pelo top $20\%$ de visualizações).

Sob a ótica da metodologia **CRISP-ML(Q) + Data Storytelling**, o problema central consiste em responder a três desafios complementares:

1. **Predição e Calibração de Alcance:** Estimar a probabilidade de um vídeo cruzar a fronteira do top $20\%$ (`viral == 1`) e prever seu volume esperado de visualizações em escala logarítmica ($\ln(1 + \text{views})$).
2. **Isolamento Temporal de Regimes de Informação (*Cold Start* vs. *Early Engagement*):** Distinguir o poder preditivo disponível antes da publicação do vídeo (metadados + inscritos) do ganho marginal trazido por sinais relativos de engajamento pós-upload, blindando o modelo contra vazamento de dados (*target* e *proxy leakage*).
3. **Desconfundimento Estrutural (*Deconfounding*):** Separar a inércia do porte do canal (`subscriber_count` e `category`) das escolhas táticas controladas pelo criador (`thumbnail_face`, `tags_count`, `duration_sec`, `title_length`, `description_length`, `upload_hour`, `upload_day`).

> [!IMPORTANT]
> **Escopo e Limites do Dataset Sintético (`datasheet.md`):** A base foi gerada sinteticamente em Python (licença CC0) com distribuições lognormais parametrizadas por categoria. Os achados e modelos deste projeto destinam-se a validação metodológica, engenharia de *Machine Learning* e *Data Storytelling*, não devendo subsidiar decisões comerciais reais sem re-treinamento em telemetria proprietária da plataforma.

---

## 2. Auditoria Física e Profiling Estatístico (*Data Understanding / Prepare*)

A inspeção *read-only* sobre `/home/tales/Documentos/github/youtube_viral_video_prediction/viral_videos.csv` confirmou integridade estrutural completa:

- **Dimensões Exatas:** $12.000$ registros $\times$ $15$ colunas.
- **Valores Ausentes (`null` / `NaN` / vazios):** $0$ em todas as $15$ colunas.
- **Duplicatas:** $0$ linhas duplicadas; $12.000$ identificadores únicos (`vid_000001` a `vid_012000`).

### 2.1. Catálogo de Variáveis e Estatísticas de Distribuição ($N = 12.000$)

| Coluna | Tipo | Grupo Funcional | Mín | $P_{25}$ | Mediana ($P_{50}$) | Média | Desvio-Padrão | $P_{75}$ | $P_{95}$ | Máx | Skewness |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `video_id` | `str` | Identificador (`vid_XXXXXX`) | — | — | — | — | — | — | — | — | — |
| `category` | `str` | Categoria ($15$ classes) | — | — | — | — | — | — | — | — | — |
| `upload_day` | `str` | Dia da semana ($7$ classes) | — | — | — | — | — | — | — | — | — |
| `duration_sec` | `int64` | Metadado Pré-Upload | $30$ | $408$ | $614$ | $631,6$ | $285,8$ | $822$ | $1.132$ | $1.960$ | $+0,38$ |
| `title_length` | `int64` | Metadado Pré-Upload | $10$ | $45$ | $55$ | $54,5$ | $15,0$ | $65$ | $79$ | $109$ | $+0,04$ |
| `thumbnail_face` | `int64` | Metadado Pré-Upload ($\{0,1\}$) | $0$ | $0$ | $0$ | $0,400$ | $0,490$ | $1$ | $1$ | $1$ | $+0,41$ |
| `upload_hour` | `int64` | Metadado Pré-Upload ($0\text{--}23$) | $0$ | $5$ | $12$ | $11,5$ | $6,9$ | $18$ | $22$ | $23$ | $-0,01$ |
| `tags_count` | `int64` | Metadado Pré-Upload | $0$ | $9$ | $14$ | $14,6$ | $7,7$ | $20$ | $27$ | $44$ | $+0,35$ |
| `description_length` | `int64` | Metadado Pré-Upload | $0$ | $300$ | $504$ | $508,9$ | $287,2$ | $709$ | $995$ | $1.618$ | $+0,29$ |
| `subscriber_count` | `int64` | Canal (Lognormal) | $3.636$ | $158.522$ | $381.382$ | $1.009.487$ | $2.585.562$ | $929.845$ | $3.639.782$ | $100.000.000$ | **$+14,47$** |
| `views` | `int64` | **Target Contínuo** | $139$ | $61.236$ | $185.268$ | $767.752$ | $3.526.908$ | $556.971$ | $2.861.486$ | $199.465.656$ | **$+29,53$** |
| `likes` | `int64` | Pós-Upload (Bruto) | $6$ | $2.790$ | $8.752$ | $37.506$ | $182.312$ | $26.571$ | $141.412$ | $11.592.871$ | $+31,42$ |
| `comments` | `int64` | Pós-Upload (Bruto) | $0$ | $317$ | $1.018$ | $4.740$ | $23.314$ | $3.207$ | $17.704$ | $1.116.186$ | $+25,84$ |
| `shares` | `int64` | Pós-Upload (Bruto) | $1$ | $642$ | $2.051$ | $9.503$ | $46.445$ | $6.527$ | $34.725$ | $2.367.174$ | $+26,19$ |
| `viral` | `int64` | **Target Binário** ($P_{80}$) | $0$ | $0$ | $0$ | $0,200$ | $0,400$ | $0$ | $1$ | $1$ | $+1,50$ |

### 2.2. Diagnóstico de Assimetria (*Skewness* & *Kurtosis*) e Normalização `log1p`

As variáveis de escala exibem cauda longa extrema na escala linear, mas convergem para distribuições gaussianas bem comportadas após a transformação $\ln(1 + x)$:

- **`views` bruto vs. `log1p(views)`:** A *skewness* cai de **$+29,532$** (curtose excessiva $= 1.305,18$) para **$+0,053$** (curtose excessiva $= +0,080$).
- **`subscriber_count` bruto vs. `log1p(subscriber_count)`:** A *skewness* cai de **$+14,472$** (curtose excessiva $= 380,74$) para **$+0,145$** (curtose excessiva $= +0,055$).

Essa propriedade comprova matematicamente a necessidade de modelar a regressão no espaço transformado $y = \ln(1 + \text{views})$ sob perda quadrática logarítmica (**RMSLE**).

### 2.3. Perfil Estatístico por Categoria Editorial ($15$ Categorias)

O dataset possui distribuição uniforme de volume entre categorias ($764$ a $825$ vídeos por categoria), mas forte estratificação na mediana de inscritos, duração e taxa de viralidade:

| Categoria | Vídeos ($n$) | Taxa `viral` (%) | Mediana `views` | Média `views` | Mediana `subscriber_count` | Mediana `duration_sec` | Mediana `views / subs` | `like_rate` Médio (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Music** | $776$ | **57,2%** | $1.016.084$ | $3.531.307$ | $2.057.548$ | $239\text{ s}$ | $0,456$ | $4,93\%$ |
| **Comedy** | $775$ | **35,2%** | $408.434$ | $1.200.731$ | $850.720$ | $297\text{ s}$ | $0,529$ | $4,91\%$ |
| **Sports** | $806$ | **27,5%** | $294.546$ | $847.503$ | $594.324$ | $475\text{ s}$ | $0,502$ | $4,84\%$ |
| **Gaming** | $807$ | **24,3%** | $238.073$ | $935.445$ | $495.660$ | $885\text{ s}$ | $0,475$ | $5,03\%$ |
| **Beauty** | $817$ | **23,4%** | $248.668$ | $724.772$ | $465.658$ | $547\text{ s}$ | $0,491$ | $5,05\%$ |
| **Tech** | $824$ | **19,1%** | $196.662$ | $697.197$ | $398.782$ | $602\text{ s}$ | $0,475$ | $5,02\%$ |
| **News** | $825$ | **18,5%** | $190.937$ | $531.972$ | $400.053$ | $367\text{ s}$ | $0,475$ | $5,08\%$ |
| **Cooking** | $794$ | **17,1%** | $175.922$ | $595.065$ | $365.019$ | $600\text{ s}$ | $0,464$ | $4,95\%$ |
| **Finance** | $821$ | **15,2%** | $163.666$ | $477.926$ | $299.502$ | $740\text{ s}$ | $0,491$ | $5,07\%$ |
| **DIY** | $779$ | **13,5%** | $136.605$ | $425.437$ | $285.334$ | $660\text{ s}$ | $0,481$ | $5,02\%$ |
| **Education** | $764$ | **12,3%** | $127.734$ | $419.567$ | $275.011$ | $885\text{ s}$ | $0,444$ | $5,05\%$ |
| **Fitness** | $776$ | **11,5%** | $125.837$ | $350.530$ | $259.620$ | $716\text{ s}$ | $0,468$ | $4,96\%$ |
| **Science** | $808$ | **11,0%** | $111.468$ | $354.748$ | $251.703$ | $850\text{ s}$ | $0,437$ | $5,14\%$ |
| **Travel** | $807$ | **8,3%** | $89.083$ | $258.448$ | $201.805$ | $891\text{ s}$ | $0,482$ | $5,06\%$ |
| **Vlog** | $821$ | **7,2%** | $94.164$ | $261.428$ | $198.270$ | $706\text{ s}$ | $0,478$ | $5,01\%$ |

**Descoberta de Confundimento (Paradoxo de Simpson):** Na análise global não condicional, `duration_sec` apresenta correlação negativa com $\ln(1 + \text{views})$ ($r = -0,2015$). Contudo, ao controlar pelo tamanho do canal usando o multiplicador relativo $\ln(\text{views}/\text{subscriber\_count})$, a correlação cai para **$r = -0,0077$ (nula)**. O aparente benefício de vídeos curtos decorre do fato de `Music` ($239\text{ s}$) e `Comedy` ($297\text{ s}$) possuírem canais com medianas de inscritos $5\text{x}$ a $10\text{x}$ superiores às categorias de vídeos longos (`Travel`, `Science`, `Education`). Em contrapartida, `thumbnail_face` ($r = +0,1112$ no multiplicador, elevando a taxa viral de $18,1\%$ para $22,8\%$ com mediana idêntica de inscritos) e `tags_count` ($r = +0,0549$) exibem efeito positivo genuíno.

---

## 3. Análise Crítica de *Data Leakage* (`Target Leakage` vs. `Proxy Leakage`)

### 3.1. *Target Leakage* Determinístico entre `views` e `viral`

A auditoria nas $12.000$ linhas ordenadas por `views` confirmou que `viral` é uma função indicadora exata do quantil $80\%$ ($P_{80}$):

- Posição $9.600$ ($80,00\%$ inferiores, `viral == 0`): $\max(\text{views} \mid \text{viral}=0) = 734.299$.
- Posição $9.601$ ($20,00\%$ superiores, `viral == 1`): $\min(\text{views} \mid \text{viral}=1) = 734.379$.

$$\text{viral}_i = \mathbb{I}(\text{views}_i \ge 734.379)$$

> [!CAUTION]
> **Bloqueio Estrito entre `views` e `viral`:** Usar `views` (ou qualquer transformação monotônica de `views`, como `log_views` ou `views / subscriber_count`) como preditor de `viral` introduz $100\%$ de *Target Leakage*, reduzindo o problema a uma regra de corte trivial.

### 3.2. *Proxy Leakage* nas Métricas Pós-Upload Brutas (`likes`, `comments`, `shares`)

A auditoria de correlações revelou que as contagens absolutas de engajamento possuem colinearidade quase unitária com `views`:

- $r(\text{views}, \text{likes}) = +0,9278 \quad \mid \quad r(\ln(1+\text{views}), \ln(1+\text{likes})) = \mathbf{+0,9736}$
- $r(\text{views}, \text{comments}) = +0,9248 \quad \mid \quad r(\ln(1+\text{views}), \ln(1+\text{comments})) = \mathbf{+0,9495}$
- $r(\text{views}, \text{shares}) = +0,8887 \quad \mid \quad r(\ln(1+\text{views}), \ln(1+\text{shares})) = \mathbf{+0,9498}$

Isso ocorre porque, no processo gerador do dataset (`datasheet.md`), `likes` é calculado como $\text{views} \times \text{like\_rate}$ (onde $\text{like\_rate} \sim U(0,02;\, 0,08)$), `comments` como fração de `likes` ($\in [0,00;\, 0,20]$) e `shares` como fração de `likes` ($\in [0,09;\, 0,40]$). Portanto:

1. **Contagens Absolutas = Proxy Leakage:** Incluir `likes`, `comments` ou `shares` brutos como preditores de `viral` ou de `views` vaza diretamente a ordem de grandeza de `views` (pois $\ln(\text{likes}) = \ln(\text{views}) + \ln(\text{like\_rate})$, onde a variância de $\ln(\text{views})$ domina mais de $94\%$ da soma).
2. **Taxas Relativas = Sinais Pós-Upload Limpos de Escala:** Ao remover `views` do numerador por meio das razões adimensionais (`like_rate = likes/views`, `comment_rate = comments/views`, `share_rate = shares/views`), a correlação com $\ln(1+\text{views})$ cai para **$r = -0,0171$, $r = -0,0004$ e $r = -0,0028$** (e com `viral` para $r \in [-0,0228;\, -0,0077]$).

### 3.3. Definição dos Regimes de Predição (Pré-Publicação vs. Pós-Publicação)

| Regime Analítico | Momento Operacional | Variáveis Permitidas na Matriz $X$ | Variáveis Estritamente Bloqueadas |
| :--- | :--- | :--- | :--- |
| **Regime A: Pré-Publicação (*Cold Start*)** | Antes do upload ($t = 0$) | `category`, `duration_sec`, `title_length`, `thumbnail_face`, `upload_hour`, `upload_day`, `tags_count`, `description_length`, `subscriber_count` (e suas transformações matemáticas puras). | `views`, `viral`, `likes`, `comments`, `shares` e quaisquer taxas de engajamento pós-upload. |
| **Regime B: Pós-Publicação (*Early Signals Relativos*)** | Após janela inicial de exposição ($t = \Delta t$) | Todas as variáveis do **Regime A** + taxas relativas adimensionais (`like_rate`, `comment_rate`, `share_rate`, `comments_per_like`, `shares_per_like`). | `views` bruto, `viral`, `likes` bruto, `comments` bruto, `shares` bruto e `views / subscriber_count`. |

---

## 4. Contratos de Qualidade de Dados (*Data Contracts*)

O pipeline de preparação (`src/youtube_viral/contracts.py`) executará automaticamente as 7 regras contratuais abaixo antes de liberar o `DataFrame` para a etapa de *Data Prep & Feature Engineering*:

| Regra ID | Dimensão da Qualidade | Especificação Formal / Invariante | Severidade |
| :--- | :--- | :--- | :--- |
| **DC-01** | Volumetria e Unicidade | `df.shape == (12000, 15)` e `df['video_id'].is_unique == True` com padrão `^vid_\d{6}$`. | Blocker (`ValueError`) |
| **DC-02** | Completude Estrita | `df.isna().sum().sum() == 0` (zero valores ausentes em todas as colunas). | Blocker (`ValueError`) |
| **DC-03** | Domínio Categórico | `set(df['category']) == 15 categorias esperadas` e `set(df['upload_day']) == {'Mon','Tue','Wed','Thu','Fri','Sat','Sun'}`. | Blocker (`ValueError`) |
| **DC-04** | Domínio de Metadados | `duration_sec` $\in [30, 3600]$; `title_length` $\in [1, 200]$; `thumbnail_face` $\in \{0, 1\}$; `upload_hour` $\in [0, 23]$; `tags_count` $\ge 0$; `description_length` $\ge 0$. | Blocker (`ValueError`) |
| **DC-05** | Invariantes de Escala e Engajamento | `subscriber_count > 0`; `views > 0`; $0 \le \text{likes} \le \text{views}$; $0 \le \text{comments} \le \text{views}$; $0 \le \text{shares} \le \text{views}$. | Blocker (`ValueError`) |
| **DC-06** | Consistência Determinística do Target | `viral` $\in \{0, 1\}$, `df['viral'].mean() == 0.20`, e $\max(\text{views} \mid \text{viral}=0) < \min(\text{views} \mid \text{viral}=1)$. | Blocker (`ValueError`) |
| **DC-07** | Barreira Anti-Leakage de Features | Nenhuma coluna em `{'video_id', 'views', 'viral', 'likes', 'comments', 'shares'}` pode estar presente em $X$; taxas relativas são bloqueadas quando `scenario == 'A'`. | Blocker (`ValueError`) |
