# Política de Proteção de Branch: `main`

Este diretório documenta a política de governança da branch `main` e fornece o script de automação (`apply.sh`) para aplicação via GitHub REST API.

---

## 1. Especificação da Política

A branch `main` é a linha de base de produção e publicação deste repositório. A política definida abaixo impede alterações diretas e exige validação de integridade por pull request antes da mesclagem.

| Configuração | Valor | Justificativa |
|:---|:---:|:---|
| **Branch Alvo** | `main` | Branch principal e base de deploy do projeto. |
| **Require status checks to pass** | `Sim` | Bloqueia mesclagem se os testes ou linter falharem. |
| **Required check (Job de CI)** | `build-and-publish` | Nome do job executado em `.github/workflows/publish.yml`. |
| **Require branches to be up to date** | `Sim` | Exige branch atualizada com a `main` antes do merge. |
| **Require pull request before merging** | `Sim` | Impede commits diretos na `main`. |
| **Required approving review count** | `1` | Exige no mínimo 1 aprovação formal de revisão. |
| **Dismiss stale reviews** | `Sim` | Invalida aprovações quando novos commits são enviados ao PR. |
| **Do not allow bypassing (Enforce Admins)** | `Sim` | Aplica as mesmas regras aos administradores do repositório. |
| **Allow force pushes** | `Não` | Bloqueia regravação de histórico (`git push --force`). |
| **Allow deletions** | `Não` | Impede a exclusão acidental da branch `main`. |

> **Nota de Divergência Técnica:** O enunciado padrão previa o check com o nome `publish`. No arquivo de workflow real (`.github/workflows/publish.yml`), o identificador do job é `build-and-publish`. A política foi ajustada para utilizar exatamente o contexto real `build-and-publish`.

---

## 2. Aplicação Automatizada

O script `apply.sh` aplica a política de forma idempotente utilizando a CLI oficial `gh` ou `curl` com autenticação via token:

```bash
# Execução via GitHub CLI (autenticado)
bash .github/branch-protection/apply.sh --repo talesgomes27/youtube_viral_video_prediction

# Execução via token de ambiente
GITHUB_TOKEN="ghp_xxx" bash .github/branch-protection/apply.sh --repo talesgomes27/youtube_viral_video_prediction
```

---

## 3. Requisitos de Permissão

Para aplicar a política, a conta ou token deve possuir permissão de administração no repositório (`admin:repo`).
