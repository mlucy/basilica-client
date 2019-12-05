set +eux
python setup.py sdist
python -m twine upload dist/*
