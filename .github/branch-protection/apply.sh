#!/usr/bin/env bash
# ==============================================================================
# Script de automação idempotente para proteção da branch 'main'
# Repositório: youtube_viral_video_prediction
# ==============================================================================

set -euo pipefail

TARGET_REPO="${GITHUB_REPOSITORY:-}"

# Parse de argumentos
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      TARGET_REPO="$2"
      shift 2
      ;;
    -h|--help)
      echo "Uso: $0 [--repo owner/name]"
      exit 0
      ;;
    *)
      echo "Opção desconhecida: $1" >&2
      exit 1
      ;;
  esac
done

# Fallback: tentar inferir a partir do remote origin do git
if [[ -z "$TARGET_REPO" ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
    if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+/[^/.]+)(\.git)?$ ]]; then
      TARGET_REPO="${BASH_REMATCH[1]}"
    fi
  fi
fi

if [[ -z "$TARGET_REPO" ]]; then
  echo "Erro: Repositório não informado. Use --repo owner/name ou defina GITHUB_REPOSITORY." >&2
  exit 1
fi

echo "==> Configurando proteção de branch para: ${TARGET_REPO} (branch: main)"

# Payload canônico de proteção
PAYLOAD='{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "build-and-publish"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}'

# Método 1: GitHub CLI (gh)
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "==> Utilizando GitHub CLI (gh) para aplicar a proteção..."
  echo "$PAYLOAD" | gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${TARGET_REPO}/branches/main/protection" \
    --input - >/dev/null

  echo "==> Proteção aplicada com sucesso. Configuração confirmada:"
  gh api \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${TARGET_REPO}/branches/main/protection"

# Método 2: curl com GITHUB_TOKEN ou GH_TOKEN
elif [[ -n "${GITHUB_TOKEN:-}" ]] || [[ -n "${GH_TOKEN:-}" ]]; then
  TOKEN="${GITHUB_TOKEN:-${GH_TOKEN}}"
  echo "==> Utilizando curl com token de autenticação..."
  HTTP_CODE="$(curl -s -o /tmp/branch_protection_resp.json -w "%{http_code}" \
    -X PUT \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${TARGET_REPO}/branches/main/protection" \
    -d "$PAYLOAD")"

  if [[ "$HTTP_CODE" -lt 200 || "$HTTP_CODE" -ge 300 ]]; then
    echo "Erro: Falha na API do GitHub (HTTP $HTTP_CODE):" >&2
    cat /tmp/branch_protection_resp.json >&2
    exit 1
  fi

  echo "==> Proteção aplicada com sucesso. Configuração confirmada:"
  curl -s \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${TARGET_REPO}/branches/main/protection"

else
  echo "Erro: Autenticação GitHub não encontrada." >&2
  echo "Para aplicar a proteção de branch:" >&2
  echo "  1. Instale e autentique a CLI do GitHub: 'gh auth login' com escopo 'admin:repo'; OU" >&2
  echo "  2. Defina a variável de ambiente: export GITHUB_TOKEN=\"ghp_seu_token_admin\"" >&2
  exit 1
fi
