#!/bin/bash
set -euo pipefail

for bench in */; do
  dirname="$(basename "${bench}")"
  cd $bench
  rm -f compile_commands.json
  bear make clang
  cd -
done