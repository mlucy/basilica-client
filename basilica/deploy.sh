set +eux
python setup.py sdist
twine upload dist/*
