.PHONY: install clean test dev train format lint help logs clean-logs clean-tuned-gp clean-tuned-kan

ENV_NAME = GP_sol
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "")
CONDA_ACTIVATE := $(shell if [ -f "$(CONDA_BASE)/etc/profile.d/conda.sh" ]; then echo "$(CONDA_BASE)/etc/profile.d/conda.sh"; elif [ -f "$(CONDA_BASE)/Scripts/activate" ]; then echo "$(CONDA_BASE)/Scripts/activate"; else echo ""; fi)

help:
	@echo "Available targets:"
	@echo "  install  - Set up conda environment and install dependencies"
	@echo "  clean    - Remove conda environment"
	@echo "  test     - Run tests with coverage"
	@echo "  dev      - Start development session"
	@echo "  run-gp   - Run GP model in tmux"
	@echo "  run-kan  - Run KAN model in tmux"
	@echo "  format   - Format code"
	@echo "  lint     - Run linting"
	@echo "  logs     - Display log files"
	@echo "  clean-logs - Delete log files"
	@echo "  clean-tuned-gp - Delete GP tuned files (.ini and .pkl)"
	@echo "  clean-tuned-kan - Delete KAN tuned files (.ini and .pkl)"
	@echo "  help     - Show this help"

install:
	@chmod +x scripts/init.sh
	@./scripts/init.sh

clean:
	@echo "Removing conda environment..."
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV_NAME)" ]; then \
		echo "ERROR: Your shell is the $(ENV_NAME) environment. Please run 'conda deactivate' and then run 'make clean' again."; \
		exit 1; \
	elif conda env list | grep -q "$(ENV_NAME)"; then \
		conda env remove -n $(ENV_NAME) -y; \
		echo "Environment $(ENV_NAME) removed successfully"; \
	else \
		echo "Environment $(ENV_NAME) not found"; \
	fi

define conda_run
	@if [ -n "$(CONDA_ACTIVATE)" ]; then \
		. "$(CONDA_ACTIVATE)" && conda activate $(ENV_NAME) && $(1); \
	else \
		echo "Warning: Could not find conda activation script. Trying direct activation..."; \
		conda activate $(ENV_NAME) && $(1); \
	fi
endef

test:
	$(call conda_run,python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing)

test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify FILE parameter"; \
		echo "Usage: make test-file FILE=tests/test_gp.py"; \
		exit 1; \
	fi
	$(call conda_run,python -m pytest $(FILE) -v)

dev:
	@tmux kill-session -t gp_dev 2>/dev/null || true
	@tmux new-session -d -s gp_dev -n main
	@tmux send-keys -t gp_dev:main "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME); else conda activate $(ENV_NAME); fi" Enter
	@tmux new-window -t gp_dev -n logs
	@tmux send-keys -t gp_dev:logs "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && tail -f logs/*.log; else conda activate $(ENV_NAME) && tail -f logs/*.log; fi" Enter
	@echo "Dev session ready: tmux attach-session -t gp_dev"

run-gp:
	@mkdir -p logs
	@tmux kill-session -t gp_sol_gp 2>/dev/null || true
	@tmux new-session -d -s gp_sol_gp -n gp
	@tmux send-keys -t gp_sol_gp:gp "conda activate $(ENV_NAME) && python main.py" Enter
	@tmux pipe-pane -t gp_sol_gp:gp "cat > logs/gp.log"
	@echo "GP model session ready: tmux attach-session -t gp_sol_gp"
	@echo "Log file: logs/gp.log"

run-kan:
	@mkdir -p logs
	@tmux kill-session -t gp_sol_kan 2>/dev/null || true
	@tmux new-session -d -s gp_sol_kan -n kan
	@tmux send-keys -t gp_sol_kan:kan "conda activate $(ENV_NAME) && python main_kan.py" Enter
	@tmux pipe-pane -t gp_sol_kan:kan "cat > logs/kan.log"
	@echo "KAN model session ready: tmux attach-session -t gp_sol_kan"
	@echo "Log file: logs/kan.log"

format:
	$(call conda_run,black src/ tests/ --line-length 88)
	$(call conda_run,isort src/ tests/)

lint:
	$(call conda_run,flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503)
	$(call conda_run,mypy src/ --ignore-missing-imports)

sessions:
	@echo "Active tmux sessions:"
	@tmux list-sessions 2>/dev/null || echo "No active tmux sessions"

kill-sessions:
	@echo "Killing tmux sessions..."
	@tmux kill-session -t gp_dev 2>/dev/null || echo "No dev session to kill"
	@tmux kill-session -t gp_sol_gp 2>/dev/null || echo "No GP session to kill"
	@tmux kill-session -t gp_sol_kan 2>/dev/null || echo "No KAN session to kill"
	@echo "Sessions killed!"

clean-figures:
	@echo "Cleaning up figures and logs..."
	@rm -rf logs
	@rm -f figures/*.png figures/*.gif
	@rm -f tests/figures/*.png
	@echo "Figures and logs cleaned!"

clean-config:
	@echo "Cleaning up config files..."
	@rm -f config/*.ini config/*.pkl
	@echo "Config files cleaned!"
	@echo "Note: test.ini was preserved"

clean-all: clean-figures clean-config clean-tuned-gp clean-tuned-kan
	@echo "All generated files cleaned!"

logs:
	@echo "Displaying log files from logs/ directory:"
	@if [ -d "logs" ] && [ "$(shell ls logs/*.log 2>/dev/null | wc -l)" -gt 0 ]; then \
		echo "Found log files:"; \
		ls -la logs/*.log; \
		echo ""; \
		echo "=== GP Model Log ==="; \
		if [ -f "logs/gp.log" ]; then \
			echo "Last 20 lines of gp.log:"; \
			tail -20 logs/gp.log; \
		else \
			echo "gp.log not found"; \
		fi; \
		echo ""; \
		echo "=== KAN Model Log ==="; \
		if [ -f "logs/kan.log" ]; then \
			echo "Last 20 lines of kan.log:"; \
			tail -20 logs/kan.log; \
		else \
			echo "kan.log not found"; \
		fi; \
	else \
		echo "No log files found in logs/ directory"; \
		echo "Run 'make run-gp' or 'make run-kan' to generate logs"; \
	fi

clean-logs:
	@echo "Cleaning log files..."
	@if [ -d "logs" ]; then \
		rm -f logs/*.log; \
		echo "Log files deleted from logs/ directory"; \
	else \
		echo "logs/ directory not found"; \
	fi

clean-tuned-gp:
	@echo "Cleaning GP tuned files..."
	@rm -f config/gp.ini config/gp_sigmas.pkl
	@echo "GP tuned files deleted!"

clean-tuned-kan:
	@echo "Cleaning KAN tuned files..."
	@rm -f config/gp_kan.ini config/gp_kan_params.pkl
	@echo "KAN tuned files deleted!"

info:
	@echo "Environment Information:"
	@echo "Environment: $(ENV_NAME)"
	@echo "Python version:"
	$(call conda_run,python --version)
	@echo "Installed packages:"
	$(call conda_run,pip list | head -20)
	@echo "... and more" 