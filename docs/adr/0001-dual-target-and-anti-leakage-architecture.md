# ADR-0001: Arquitetura de Modelagem Dual e Isolamento Anti-Leakage entre Cenários A e B

- **Status:** Aceito
- **Data:** 2026-10-15
- **Projeto:** `youtube_viral_video_prediction` (Arquétipo A)

---

## Contexto

Na base `viral_videos.csv` ($N = 12.000$), a variável binária `viral` é definida deterministicamente pelo quantil $P_{80}$ da variável contínua `views` ($\text{views} \ge 734.379$). Além disso, as métricas de engajamento pós-upload (`likes`, `comments`, `shares`) foram sintetizadas multiplicando `views` por taxas limitadas ($\text{like\_rate} \in [0,02;\, 0,08]$). Por fim, o tamanho do canal (`subscriber_count`) apresenta distribuição lognormal fortemente correlacionada com o volume de visualizações ($r = +0,8097$ em escala logarítmica) e varia drasticamente entre categorias editoriais, gerando confundimento (Paradoxo de Simpson) na avaliação de metadados como `duration_sec`.

Precisamos definir a arquitetura de formulação dos alvos ($y$) e das matrizes de atributos ($X$) para impedir vazamento de dados (*target leakage*) e evitar conclusões espúrias causadas pelo porte do canal.

---

## Opções Consideradas

### Opção 1: Classificação Binária Única com Todas as Colunas Exceto `views`
- **Prós:** Implementação simples com um único modelo supervisionado.
- **Contras:** Incluir `likes`, `comments` e `shares` brutos vaza diretamente a escala de `views` ($r > 0,95$ em log-escala), invalidando o modelo. Além disso, modelar apenas o alvo binário descarta a informação métrica de distância até a fronteira de $734.379$ visualizações.

### Opção 2: Regressão Única de `views` Bruto
- **Prós:** Aproveita a variável contínua original.
- **Contras:** A cauda longa extrema (até $199\text{M}$ de visualizações) faz com que a perda quadrática (MSE) seja dominada por menos de $1\%$ das observações, degradando a predição na região da fronteira de viralidade ($P_{80}$). Não isola diretamente o efeito confundidor de `subscriber_count`.

### Opção 3: Arquitetura Tripla de Alvos (Dual + Desconfundimento) com Isolamento de Cenários A e B
- **Prós:**
  1. Treina tanto classificadores diretos para `viral` quanto regressores na escala estabilizada $y = \ln(1 + \text{views})$, comparando a classificação direta com o corte em $P_{80}$ das predições contínuas.
  2. Adiciona um terceiro alvo complementar de desconfundimento, $M = \ln(\text{views} / \text{subscriber\_count})$, removendo a inércia do tamanho do canal para medir o efeito puro dos metadados do criador.
  3. Separa fisicamente as matrizes $X_A$ (*Cold Start*, apenas metadados pré-upload) e $X_B$ (*Early Signals*, adicionando apenas razões adimensionais `likes/views`, `comments/views`, `shares/views`), bloqueando contagens brutas por contrato automatizado.
- **Contras:** Exige orquestração de múltiplos estimadores e validação cruzada estratificada unificada.

---

## Decisão

Adotamos a **Opção 3**: Arquitetura de Modelagem Dual (`viral` e $\ln(1 + \text{views})$) acrescida do modelo de Desconfundimento ($\ln(\text{views} / \text{subscriber\_count})$), operando sobre matrizes de atributos estritamente isoladas entre o **Cenário A (Pré-Upload)** e o **Cenário B (Pós-Upload Relativo)**, protegidas em tempo de execução pela regra contratual `DC-07`.

---

## Consequências

### Positivas
- Eliminação matemática e programática de *target leakage* direto e indireto.
- Capacidade de provar empiricamente a Hipótese $H_5$ (comparando Cenário A vs. Cenário B nas mesmas dobras de validação).
- Desacoplamento claro entre o efeito estrutural da base de inscritos ($H_1, H_2$) e as alavancas táticas do criador ($H_3, H_4$).

### Negativas / Trade-offs
- Aumento moderado no tempo de execução da suíte de treinamento (12 combinações de modelos $\times$ cenários em 5-Fold CV), mitigado pelo tamanho compacto da base ($12.000$ linhas, tempo total estimado $< 45$ segundos em CPU).
