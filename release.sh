#!/usr/bin/env bash
set -euo pipefail

VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
TAG="v${VERSION}"

echo "Preparing release ${TAG}..."

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: Tag ${TAG} already exists. Bump the version in pyproject.toml first."
    exit 1
fi

git tag "$TAG"
git push origin "$TAG"

echo "Tagged ${TAG} and pushed. GitHub Actions will publish to PyPI."
