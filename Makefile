ENV_NAME = ligo

.PHONY: env html clean

env:
	@echo ">>> Setting up or updating Conda environment: $(ENV_NAME)"
	conda env create -f environment.yml --name $(ENV_NAME) || \
	conda env update -f environment.yml --name $(ENV_NAME)
	@echo ">>> Done. Activate with: conda activate $(ENV_NAME)"

html:
	@echo ">>> Building local HTML with MyST..."
	myst build --html
	@echo ">>> Built site under _build/html/"

clean:
	@echo ">>> Cleaning build artifacts..."
	rm -rf _build/* figures/* audio/*
	@echo ">>> Clean complete."

