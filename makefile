VENV_DIR = .venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip
PYTHON_VENV = $(VENV_DIR)/bin/python

.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: install
install:
	$(PIP) install -r requirements.txt


install-editable:
	pip install -e  ../cjlapao/pd-ai-agent-core

uninstall-editable:
	pip uninstall -y pd-ai-agent-core