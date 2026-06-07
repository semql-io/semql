default: check

fmt:
    uv run ruff format packages/

lint:
    uv run ruff check --fix packages/

typecheck:
    uv run mypy packages/
    uv run pyright packages/

test *args:
    uv run pytest {{args}}

check: fmt lint typecheck test

# Mutation testing on semql core. Results saved to .mutmut-cache.
# Show a summary with `just mutmut-results` after a run completes.
mutmut:
    uv run mutmut run

mutmut-results:
    uv run mutmut results

hooks:
    uv run pre-commit install

# ---------------------------------------------------------------------------
# Release: build + validate + publish
# ---------------------------------------------------------------------------
#
# Tokens live encrypted in ``secrets.enc.yaml`` (sops + age). The
# publish recipes decrypt on demand using your age key at
# ``~/.config/sops/age/keys.txt``. Generate one with ``age-keygen``,
# then ``sops secrets.enc.yaml`` to drop your real ``test_pypi_token``
# / ``pypi_token`` in.
#
# Override via ``UV_PUBLISH_TOKEN=... just publish`` if you want to
# bypass sops for a one-off.
#
# Recipes refuse to upload if `twine check` flags any metadata problem.
# Order matters: publish `semql` before its dependents — dependents
# resolve against the index, and PyPI takes a moment to advertise new
# versions.

# Path to the sops age key. Override if your key lives elsewhere.
sops_age_key := env_var_or_default("SOPS_AGE_KEY_FILE", env_var("HOME") + "/.config/sops/age/keys.txt")
secrets_file := "secrets.enc.yaml"

# Clean ./dist
clean-dist:
    rm -rf dist

# Build wheels + sdists for one or all packages. Defaults to all.
#   just build           # all four
#   just build semql     # just semql
build *pkgs="semql semql-mcp semql-erd semql-validate-db": clean-dist
    #!/usr/bin/env bash
    set -euo pipefail
    for pkg in {{pkgs}}; do
      echo "── build $pkg ──"
      uv build --package "$pkg"
    done

# Metadata validation — same gate PyPI applies on upload.
check-dist:
    uv run twine check dist/*

# Decrypt one token field from secrets.enc.yaml. Used by the publish
# recipes; can also be called directly:
#   just _token test_pypi_token | pbcopy
_token field:
    @SOPS_AGE_KEY_FILE={{sops_age_key}} sops -d --extract '["{{field}}"]' {{secrets_file}}

# Publish to TestPyPI. Token decrypted from secrets.enc.yaml unless
# UV_PUBLISH_TOKEN is already set in the environment.
publish-test: check-dist
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
      UV_PUBLISH_TOKEN=$(just _token test_pypi_token)
      export UV_PUBLISH_TOKEN
    fi
    uv publish --publish-url https://test.pypi.org/legacy/ dist/*

# Publish to real PyPI. Token decrypted from secrets.enc.yaml unless
# UV_PUBLISH_TOKEN is already set in the environment.
publish: check-dist
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
      UV_PUBLISH_TOKEN=$(just _token pypi_token)
      export UV_PUBLISH_TOKEN
    fi
    uv publish dist/*

# End-to-end: build everything, validate, publish to TestPyPI.
release-test: (build) check-dist publish-test

# End-to-end: build everything, validate, publish to PyPI.
release: (build) check-dist publish

# Build + check a single package and stage it for a focused publish.
#   just stage semql        # leaves only semql artifacts in dist/
stage pkg: clean-dist
    uv build --package {{pkg}}
    uv run twine check dist/*

# ---------------------------------------------------------------------------
# Staged release — semql first, wait for the index, then dependents.
# ---------------------------------------------------------------------------
#
# Use the staged variants when you want to verify `semql` works on the
# target index before its dependents go out. Faster path: `release-test`
# / `release` (above) publish all four at once.

# Read a package's version straight from its pyproject.toml.
_pkg-version pkg:
    @python3 -c "import tomllib; print(tomllib.loads(open('packages/{{pkg}}/pyproject.toml').read())['project']['version'])"

# Poll TestPyPI's simple index until <pkg>=<version> is resolvable.
# Fails after ~5 minutes if the package never appears.
#   just wait-indexed-test semql
wait-indexed-test pkg:
    #!/usr/bin/env bash
    set -euo pipefail
    version=$(just _pkg-version {{pkg}})
    echo "Polling TestPyPI simple index for {{pkg}}==$version..."
    for i in $(seq 1 60); do
      if curl -sf "https://test.pypi.org/simple/{{pkg}}/" 2>/dev/null \
           | grep -q "{{pkg}}-${version}-"; then
        echo "✓ {{pkg}}==$version indexed."
        exit 0
      fi
      printf "."
      sleep 5
    done
    echo
    echo "ERROR: {{pkg}}==$version not on TestPyPI after 5 minutes." >&2
    exit 1

# Poll real PyPI's simple index until <pkg>=<version> is resolvable.
wait-indexed pkg:
    #!/usr/bin/env bash
    set -euo pipefail
    version=$(just _pkg-version {{pkg}})
    echo "Polling PyPI simple index for {{pkg}}==$version..."
    for i in $(seq 1 60); do
      if curl -sf "https://pypi.org/simple/{{pkg}}/" 2>/dev/null \
           | grep -q "{{pkg}}-${version}-"; then
        echo "✓ {{pkg}}==$version indexed."
        exit 0
      fi
      printf "."
      sleep 5
    done
    echo
    echo "ERROR: {{pkg}}==$version not on PyPI after 5 minutes." >&2
    exit 1

# Build + publish the three semql dependents to TestPyPI.
# Assumes semql is already on TestPyPI (use wait-indexed-test before).
publish-test-rest: clean-dist
    #!/usr/bin/env bash
    set -euo pipefail
    for pkg in semql-mcp semql-erd semql-validate-db; do
      echo "── build $pkg ──"
      uv build --package "$pkg"
    done
    uv run twine check dist/*
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
      UV_PUBLISH_TOKEN=$(just _token test_pypi_token)
      export UV_PUBLISH_TOKEN
    fi
    uv publish --publish-url https://test.pypi.org/legacy/ dist/*

# Build + publish the three semql dependents to real PyPI.
# Assumes semql is already on PyPI (use wait-indexed before).
publish-rest: clean-dist
    #!/usr/bin/env bash
    set -euo pipefail
    for pkg in semql-mcp semql-erd semql-validate-db; do
      echo "── build $pkg ──"
      uv build --package "$pkg"
    done
    uv run twine check dist/*
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
      UV_PUBLISH_TOKEN=$(just _token pypi_token)
      export UV_PUBLISH_TOKEN
    fi
    uv publish dist/*

# Full staged TestPyPI release: semql → wait → dependents.
release-test-staged:
    just stage semql
    just publish-test
    just wait-indexed-test semql
    just publish-test-rest

# Full staged PyPI release: semql → wait → dependents.
release-staged:
    just stage semql
    just publish
    just wait-indexed semql
    just publish-rest
