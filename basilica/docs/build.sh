mkdir -p _build
pip install ../
sphinx-build -b html . _build
