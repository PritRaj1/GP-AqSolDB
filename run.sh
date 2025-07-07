#!/bin/bash

source activate base
conda activate GP_sol

# tmux new-session -d -s GP_sol_main "python main.py"
# tmux pipe-pane -t GP_sol_main "cat > main.log"

# echo "GP_sol session started in tmux. Check main.log for output."
# echo "To attach to session: tmux attach -t GP_sol_main"
# echo "To detach from session: Ctrl+B, then D"

tmux new-session -d -s GP_sol_main "python main_kan.py"
tmux pipe-pane -t GP_sol_main "cat > main_kan.log"

echo "GP_sol session started in tmux. Check main_kan.log for output."
echo "To attach to session: tmux attach -t GP_sol_main"
echo "To detach from session: Ctrl+B, then D"
