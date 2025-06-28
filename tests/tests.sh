#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate GP_sol

> test.log

test_files=$(find tests -name "test_*.py" -type f)

if [ -z "$test_files" ]; then
    echo "No test files found in tests directory" | tee -a test.log
    exit 1
fi

for test_file in $test_files; do
    echo "Running $test_file..." | tee -a test.log
    if pytest "$test_file" 2>&1 | tee -a test.log; then
        echo "✓ $test_file completed successfully" | tee -a test.log
    else
        echo "✗ $test_file failed" | tee -a test.log
    fi
    echo "----------------------------------------" | tee -a test.log
done

echo "All tests completed. Check test.log for detailed results."
