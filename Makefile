.DEFAULT_GOAL := help

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m 

ENV_NAME := GP_sol
PYTHON := python

help: 
	@echo "$(BLUE)=== Gaussian Process AqSolDB ===$(NC)"
	@echo ""
	@echo "$(YELLOW)📦 Setup & Installation:$(NC)"
	@echo "  $(GREEN)install$(NC)         Create conda environment and install dependencies"
	@echo "  $(GREEN)uninstall$(NC)       Remove the conda environment"
	@echo ""
	@echo "$(YELLOW)🧪 Testing:$(NC)"
	@echo "  $(GREEN)test$(NC)            Run all tests"
	@echo "  $(GREEN)test-file$(NC)       Run a specific test file (FILE=path/to/test.py)"
	@echo ""
	@echo "$(YELLOW)🚀 Running Models:$(NC)"
	@echo "  $(GREEN)run-gp$(NC)          Run standard GP model in tmux (logs to logs/gp.log)"
	@echo "  $(GREEN)run-kan$(NC)         Run KAN model in tmux (logs to logs/kan.log)"
	@echo ""
	@echo "$(YELLOW)📋 Session Management:$(NC)"
	@echo "  $(GREEN)sessions$(NC)        List active tmux sessions"
	@echo "  $(GREEN)kill-sessions$(NC)   Kill all project tmux sessions"
	@echo ""
	@echo "$(YELLOW)🧹 Cleaning:$(NC)"
	@echo "  $(GREEN)clean$(NC)           Show cleaning options"
	@echo "  $(GREEN)clean-figures$(NC)   Remove figures and logs"
	@echo "  $(GREEN)clean-config$(NC)    Remove config files (.ini/.pkl, preserves test.ini)"
	@echo "  $(GREEN)clean-all$(NC)       Remove all generated files"
	@echo ""
	@echo "$(YELLOW)🔧 Development:$(NC)"
	@echo "  $(GREEN)info$(NC)            Show environment information"
	@echo "  $(GREEN)format$(NC)          Format code with black"
	@echo "  $(GREEN)lint$(NC)            Run linting with flake8"
	@echo "  $(GREEN)dev-setup$(NC)       Complete development setup"
	@echo "  $(GREEN)tree$(NC)            Show project structure"
	@echo ""
	@echo "$(YELLOW)💡 Quick Start:$(NC)"
	@echo "  1. make install    # First time setup"
	@echo "  2. make test       # Verify installation"
	@echo "  3. make run-kan    # Run the KAN model"
	@echo "  4. make sessions   # Check running sessions"

check-conda: 
	@command -v conda >/dev/null 2>&1 || { \
		echo "$(RED)Error: conda is not installed or not in PATH$(NC)"; \
		echo "Please install Anaconda or Miniconda first:"; \
		echo "  https://docs.conda.io/en/latest/miniconda.html"; \
		echo "Or add conda to your PATH manually."; \
		exit 1; \
	}

CONDA_PATH := $(shell if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then echo "~/anaconda3/etc/profile.d/conda.sh"; elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then echo "~/miniconda3/etc/profile.d/conda.sh"; else echo ""; fi)

check-env: check-conda 
	@if [ -z "$(CONDA_PATH)" ]; then \
		echo "$(RED)Error: Could not find conda installation$(NC)"; \
		echo "Please install Anaconda or Miniconda first:"; \
		echo "  https://docs.conda.io/en/latest/miniconda.html"; \
		exit 1; \
	fi
	@. $(CONDA_PATH) && \
	if ! conda env list | grep -q '$(ENV_NAME)'; then \
		echo "$(YELLOW)Environment $(ENV_NAME) not found. Run 'make install' first.$(NC)"; \
		exit 1; \
	fi

install: check-conda 
	@if [ -z "$(CONDA_PATH)" ]; then \
		echo "$(RED)Error: Could not find conda installation$(NC)"; \
		echo "Please install Anaconda or Miniconda first:"; \
		echo "  https://docs.conda.io/en/latest/miniconda.html"; \
		exit 1; \
	fi
	@echo "$(BLUE)Installing conda environment and dependencies...$(NC)"
	@. $(CONDA_PATH) && \
	if conda env list | grep -q '$(ENV_NAME)'; then \
		echo "$(YELLOW)Environment $(ENV_NAME) already exists. Updating...$(NC)"; \
	else \
		echo "$(GREEN)Creating conda environment $(ENV_NAME)...$(NC)"; \
		conda create -n $(ENV_NAME) python=3.11 -y; \
	fi
	@echo "$(GREEN)Installing tmux...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && conda install -c conda-forge tmux -y
	@echo "$(GREEN)Installing Python requirements...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && $(PYTHON) setup/requirements.py
	@echo "$(GREEN)Installation complete!$(NC)"
	@echo "$(YELLOW)To activate the environment: conda activate $(ENV_NAME)$(NC)"



test: check-env 
	@echo "$(BLUE)Running tests...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && pytest tests -v
	@echo "$(GREEN)Tests completed!$(NC)"

test-file: check-env 
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: Please specify FILE parameter$(NC)"; \
		echo "Usage: make test-file FILE=tests/test_gp.py"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running test file: $(FILE)$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && pytest $(FILE) -v

dirs:
	@mkdir -p logs

run-gp: check-env dirs 
	@echo "$(BLUE)Starting GP model in tmux session...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && tmux new-session -d -s GP_sol_gp "$(PYTHON) main.py"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && tmux pipe-pane -t GP_sol_gp "cat > logs/gp.log"
	@echo "$(GREEN)GP model started in tmux session 'GP_sol_gp'$(NC)"
	@echo ""
	@echo "$(YELLOW)--- tmux usage tips ---$(NC)"
	@echo "Attach:   tmux attach -t GP_sol_gp"
	@echo "Detach:   Press Ctrl+B, then D"
	@echo "View log: tail -f logs/gp.log"
	@echo "Kill:     tmux kill-session -t GP_sol_gp"
	@echo "List:     tmux list-sessions or make sessions"
	@echo "------------------------"
	@echo "Log file: logs/gp.log"

run-kan: check-env dirs 
	@echo "$(BLUE)Starting KAN model in tmux session...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && tmux new-session -d -s GP_sol_kan "$(PYTHON) main_kan.py"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && tmux pipe-pane -t GP_sol_kan "cat > logs/kan.log"
	@echo "$(GREEN)KAN model started in tmux session 'GP_sol_kan'$(NC)"
	@echo ""
	@echo "$(YELLOW)--- tmux usage tips ---$(NC)"
	@echo "Attach:   tmux attach -t GP_sol_kan"
	@echo "Detach:   Press Ctrl+B, then D"
	@echo "View log: tail -f logs/kan.log"
	@echo "Kill:     tmux kill-session -t GP_sol_kan"
	@echo "List:     tmux list-sessions or make sessions"
	@echo "------------------------"
	@echo "Log file: logs/kan.log"

sessions: 
	@echo "$(BLUE)Active tmux sessions:$(NC)"
	@tmux list-sessions 2>/dev/null || echo "$(YELLOW)No active tmux sessions$(NC)"

kill-sessions: 
	@echo "$(BLUE)Killing tmux sessions...$(NC)"
	@tmux kill-session -t GP_sol_gp 2>/dev/null || echo "$(YELLOW)No GP session to kill$(NC)"
	@tmux kill-session -t GP_sol_kan 2>/dev/null || echo "$(YELLOW)No KAN session to kill$(NC)"
	@tmux kill-session -t GP_sol_main 2>/dev/null || echo "$(YELLOW)No main session to kill$(NC)"
	@echo "$(GREEN)Sessions killed!$(NC)"

clean: 
	@echo "$(BLUE)Cleaning up generated files...$(NC)"
	@echo "$(YELLOW)Use 'make clean-figures' to remove figures and logs$(NC)"
	@echo "$(YELLOW)Use 'make clean-config' to remove config files (.ini/.pkl)$(NC)"
	@echo "$(YELLOW)Use 'make clean-all' to remove everything$(NC)"

clean-figures: 
	@echo "$(BLUE)Cleaning up figures and logs...$(NC)"
	@rm -rf logs
	@rm -f figures/*.png figures/*.gif
	@rm -f tests/figures/*.png
	@echo "$(GREEN)Figures and logs cleaned!$(NC)"

clean-config: 
	@echo "$(BLUE)Cleaning up config files...$(NC)"
	@rm -f config/*.ini config/*.pkl
	@echo "$(GREEN)Config files cleaned!$(NC)"
	@echo "$(YELLOW)Note: test.ini was preserved$(NC)"

clean-all: clean-figures clean-config
	@echo "$(GREEN)All generated files cleaned!$(NC)"

uninstall: 
	@echo "$(RED)Removing conda environment $(ENV_NAME)...$(NC)"
	@. $(CONDA_PATH) && conda env remove -n $(ENV_NAME) -y
	@echo "$(GREEN)Environment removed!$(NC)"

info: check-env 
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "$(YELLOW)Environment:$(NC) $(ENV_NAME)"
	@echo "$(YELLOW)Python version:$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && $(PYTHON) --version
	@echo "$(YELLOW)Installed packages:$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && pip list | head -20
	@echo "$(YELLOW)... and more$(NC)"

format: check-env 
	@echo "$(BLUE)Formatting code...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && black src/ tests/ --line-length 88 2>/dev/null || echo "$(YELLOW)black not installed, skipping formatting$(NC)"

lint: check-env 
	@echo "$(BLUE)Running linting...$(NC)"
	@. $(CONDA_PATH) && conda activate $(ENV_NAME) && flake8 src/ tests/ 2>/dev/null || echo "$(YELLOW)flake8 not installed, skipping linting$(NC)"

dev-setup: install format lint 
	@echo "$(GREEN)Development setup complete!$(NC)"

tree: 
	@echo "$(BLUE)Project Structure:$(NC)"
	@tree -I '__pycache__|*.pyc|*.log|*.png|*.gif' -a

.PHONY: help install test test-file run-gp run-kan sessions kill-sessions clean clean-figures clean-config clean-all uninstall info format lint dev-setup tree check-conda check-env 