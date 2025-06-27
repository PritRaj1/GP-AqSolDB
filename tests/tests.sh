#!/bin/bash

conda activate GP_sol

> test.log

test_files=$(ls "tests/test_*.py")

for test_file in $test_files; do
    echo "Running $test_file..." | tee -a test.log
    pytest $test_file 2>&1 | tee -a test.log
    echo "----------------------------------------" | tee -a test.log
done

echo "All tests completed."
