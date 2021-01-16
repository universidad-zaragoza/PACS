#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

problems_dir="${script_dir}/../problems"

pandoc --pdf-engine=xelatex "${problems_dir}/problems.md" -f markdown+smart \
    -s -o "${problems_dir}/problems.pdf" -V colorlinks --table-of-contents
