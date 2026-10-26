#!/usr/bin/env bash
# Run with the bundled PostgreSQL-16 db-tools image. Never source .env or eval input.
set -euo pipefail
umask 077
if [[ $# == 0 || ${1:-} == plan ]]; then
  printf '%s\n' 'PLAN ONLY: stop API/worker, backup v7, restore trusted archive into NEW EMPTY DB; quarantine until review.'
  exit 0
fi
die() { printf '%s\n' 'Backup/restore rejected; check explicit confirmations and trusted files safely' >&2; exit 2; }
[[ $# -ge 5 ]] || die
operation=$1; name=$2; shift 2
[[ $operation == backup || $operation == restore ]] || die
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*\.dump$ && $name != *..* ]] || die
stopped=false; trusted=false; expected=''
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-services-stopped) stopped=true; shift ;;
    --confirm-database) [[ $# -ge 2 ]] || die; expected=$2; shift 2 ;;
    --trust-local-backup) trusted=true; shift ;;
    *) die ;;
  esac
done
[[ $stopped == true && -n $expected && ${PGDATABASE:-} == "$expected" && -n ${PGHOST:-} && -n ${PGUSER:-} ]] || die
[[ $operation != restore || $trusted == true ]] || die
export PGCONNECT_TIMEOUT=3
export PGOPTIONS='-c search_path=public,pg_catalog -c statement_timeout=600000'
backup_dir=${BACKUP_DIR:-/backups}
mkdir -p -- "$backup_dir"
work=$(mktemp -d "$backup_dir/.work.XXXXXXXX")
locked=false
cleanup() { rm -rf -- "$work"; if [[ $locked == true ]]; then rmdir -- "$backup_dir/.$name.lock"; fi; }
trap cleanup EXIT
trap 'printf "%s\n" "Database tool failed or outcome unknown; no raw errors logged" >&2' ERR
mkdir -- "$backup_dir/.$name.lock" 2>"$work/errors" || die
locked=true
file="$backup_dir/$name"
# SQL has no interpolated caller arguments; comparison is done in the shell.
actual=$(psql -X -Atq -v ON_ERROR_STOP=1 -c 'SELECT current_database()' 2>"$work/errors")
[[ $actual == "$expected" ]] || die
other=$(psql -X -Atq -v ON_ERROR_STOP=1 -c "SELECT count(*) FROM pg_namespace WHERE nspname NOT IN ('public','information_schema') AND nspname NOT LIKE 'pg_%'" 2>"$work/errors")
[[ $other == 0 ]] || die
if [[ $operation == backup ]]; then
  [[ ! -e $file && ! -L $file && ! -e $file.sha256 && ! -L $file.sha256 ]] || die
  has_guard=$(psql -X -Atq -v ON_ERROR_STOP=1 -c "SELECT to_regclass('public.application_recovery_state') IS NOT NULL" 2>"$work/errors")
  [[ $has_guard == t ]] || die
  pg_dump --format=custom --no-owner --no-acl --file="$work/archive" 2>"$work/errors"
  digest=$(sha256sum "$work/archive"); digest=${digest%% *}
  printf '%s\n' "$digest" > "$work/hash"
  # Hard links are atomic no-clobber, on the same backup filesystem.
  ln -- "$work/archive" "$file" 2>"$work/errors"
  ln -- "$work/hash" "$file.sha256" 2>"$work/errors"
  printf '%s\n' 'Backup complete; protect archive/hash as sensitive data'
else
  [[ -f $file && ! -L $file && -f $file.sha256 && ! -L $file.sha256 ]] || die
  read -r expected_hash < "$file.sha256"
  [[ $expected_hash =~ ^[a-f0-9]{64}$ ]] || die
  digest=$(sha256sum "$file"); digest=${digest%% *}
  [[ $digest == "$expected_hash" ]] || die
  count=$(psql -X -Atq -v ON_ERROR_STOP=1 -c "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','S','f')" 2>"$work/errors")
  [[ $count == 0 ]] || die
  # Decode fully before touching target; restore+guard are one psql transaction.
  pg_restore --no-owner --no-acl --file="$work/restore.sql" "$file" 2>"$work/errors"
  printf '\nSET search_path=public,pg_catalog;\n' >> "$work/restore.sql"
  root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  cat "$root/deploy/restore-quarantine.sql" >> "$work/restore.sql"
  psql -X -q --single-transaction -v ON_ERROR_STOP=1 --file="$work/restore.sql" >"$work/output" 2>"$work/errors"
  printf '%s\n' 'Restore complete in quarantine; sessions revoked, budgets blocked, pending jobs failed; manual review required'
fi
