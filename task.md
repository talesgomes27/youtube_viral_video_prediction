# Plano de Rastreamento do Projeto (`task.md`)

- **Projeto:** `youtube_viral_video_prediction`
- **Arquétipo:** A — Data Science, Analytics & ML (`python-uv-quarto`)
- **Metodologia:** Modelo Híbrido CRISP-ML(Q) + Data Storytelling em 6 Seções
- **Pipeline Ativo:** `release-and-ops-pipeline` (v2.1)
- **Última Atualização:** 2026-10-15

---

## Status dos Gates de Planejamento (`pre-code-pipeline`)

- `[x]` **Gate 1: Contexto Inicial, Descoberta e Contratos de Dados (*Data Understanding & Contracts / Prepare*)** — *Aprovado*
- `[x]` **Gate 2: Ideação, Escopo, Hipóteses e Especificação (*Business Understanding, Hypotheses & Specs / Ask & Plan*)** — *Aprovado*
- `[x]` **Gate 3: Arquitetura do Pipeline: Data Prep, Modeling, Evaluation & Storytelling (*Process, Construct & Share*)** — *Aprovado*
- `[x]` **Gate 4: Scaffolding, Estrutura de Pastas e Convenções** — *Aprovado*
- `[x]` **Gate 5: Plano de Implementação, Validação & Action (*Deployment / Act*)** — *Aprovado*

---

## Status dos Gates de Execução (`execution-pipeline` — Arquétipo A)

- `[x]` **Exec Gate 0 & 1: Bootstrap de Ambiente, Contratos de Dados e Feature Engineering (TDD)** — *Aprovado*
- `[x]` **Exec Gate 2 & 3: Testes Estatísticos ($H_1\text{--}H_5$), Modelagem Tripla, Explicabilidade (`gain`/SHAP) e Figuras 300 DPI** — *Aprovado*
- `[x]` **Exec Gate 4 & 5: Relatório Quarto nas 6 Seções CRISP-ML(Q) + Storytelling, README e Auditoria Final (`test-guard` & `docs-guard`)** — *Aprovado*

---

## Status dos Gates de Release & Ops (`release-and-ops-pipeline` — Arquétipo A)

- `[x]` **Release Gate 1: Saneamento de Repositório, Segurança Pré-Push & Lockfiles** — *Aprovado*
  - `[x]` Inicialização do repositório Git (`main`) e blindagem do `.gitignore` (exceção autorizada para `!viral_videos.csv` e `!reports/metrics/*.csv`; `/_freeze/` rastreado)
  - `[x]` Varredura contra segredos/credenciais expostas e arquivos `> 50 MB`
  - `[x]` Verificação de integridade e sincronia de `pyproject.toml` e `uv.lock`
  - `[x]` Aprovação humana do Gate 1
- `[/]` **Release Gate 2: Governança Git, Versionamento Semântico (`v0.1.0`) & `CHANGELOG.md`** — *Em revisão humana*
  - `[x]` Definição da versão SemVer inicial `v0.1.0` (alinhada ao `pyproject.toml`)
  - `[x]` Redação do `CHANGELOG.md` no padrão *Keep a Changelog* (`writing-pipeline` + `humanizer`)
  - `[x]` Estruturação de commits atômicos locais (Conventional Commits) e tag anotada `v0.1.0`
  - `[ ]` Aprovação humana do Gate 2
- `[ ]` **Release Gate 3: Automação de CI/CD — GitHub Actions (`.github/workflows/publish.yml`)** — *Pendente*
- `[ ]` **Release Gate 4: Deploy, Publicação e Empacotamento (GitHub Pages)** — *Pendente*
- `[ ]` **Release Gate 5: Smoke Tests Pós-Deploy & Homologação Final** — *Pendente*
