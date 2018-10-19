mkdir -p _build _static _templates
pip install ..
sphinx-build -b html . _build
