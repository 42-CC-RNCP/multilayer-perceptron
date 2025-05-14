NAME = multlayer-perceptron
SCRIPTS_DIR = "$(shell pwd)/scripts"

# Define the default target
.PHONY: all re clean fclean re


$(NAME): all
# if poetry.lock exists, install dependencies;
# 	otherwise, run the install script
	@if [ -f poetry.lock ]; then \
		echo "Installing dependencies..."; \
		poetry update; \
	else \
		echo "poetry.lock not found. Running install script..."; \
		$(SCRIPTS_DIR)/install.sh; \
	fi
	@echo "All dependencies installed successfully."

all: $(NAME)

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type f -name "*.npz" -exec rm -rf {} +
	@find . -type f -name "*.npy" -exec rm -rf {} +
	@find . -type d -name "trained" -exec rm -rf {} +
	@find . -type d -empty -delete

fclean: clean
	@echo "Removing virtual environment..."
	@rm poetry.lock 2>/dev/null || true
	@poetry env remove
	@echo "Reseted the environment successfully."

re: clean all

.PHONY: all clean fclean re
