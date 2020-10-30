set -euo pipefail

for bench in ./rodinia/cuda/*; do
  dirname="$(basename "${bench}")"
  cd $bench
  bear make
  cd -
done