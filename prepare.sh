#!/bin/bash
set -euo pipefail

cd rodinia && /bin/bash prepare.sh && cd -
cd simple && /bin/bash prepare.sh && cd -