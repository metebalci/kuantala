@echo off
setlocal

for /f "delims=" %%v in ('python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"') do set VERSION=%%v
set TAG=v%VERSION%

echo Preparing release %TAG%...

git rev-parse %TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo Error: Tag %TAG% already exists. Bump the version in pyproject.toml first.
    exit /b 1
)

git tag %TAG%
git push origin %TAG%

echo Tagged %TAG% and pushed. GitHub Actions will publish to PyPI.
