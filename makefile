SHELL = /usr/bin/env bash

all:
	python -m compileall .

clean:
	@rm -f .coverage
	@rm -rf htmlcov
	@find . -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete
Clean: clean

pep8:
	pep8 . --ignore=E402,E501,E731

pylint:
	pylint tail; find bin -name '*.py' -exec pylint {} +

pylintE:
	pylint -E tail; find bin -name '*.py' -exec pylint -E {} +
