.PHONY: doc

export PYTHONPATH := ${PYTHONPATH}:${PWD}

# Local

call-pred:
	python3 src/application/predict_from_served_model.py

train-local:
	python3 src/application/train_local.py

# Tests

k6:
	# docker-compose run k6 run /scripts/script.js
	docker-compose run k6 run /scripts/load_test_1.js --summary-export=AKS_load_test_1.json

coverage: ## to check how many lines of codes are tested
	coverage run --source=src/ -m unittest discover -s tests/unit_tests
	coverage report -m

safety: ## to check the safety of packages / NEED AN INTERNET CONNECTION
	safety check

lint: ## to check that your code follows PEP8 standards
	flake8 src/
	flake8 tests/

isort: # to sort imports
	isort src/
	isort tests/

# Install

install: environment.yml
	conda env create -n cenv -f environment.yml

env-export: ## export virtual environment
	conda env export > environment.yml --no-builds

freeze:
	pip freeze | grep -v "pkg-resources" > requirements.txt

doc: ## to generate Sphinx documentation
	rm -rf doc/source/generated
	sphinx-apidoc -d 1 --no-toc --separate --force --private -o doc/source/generated . ./tests
	cd doc; make html