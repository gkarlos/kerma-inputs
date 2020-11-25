#!/bin/bash
set -euo pipefail

cd polybench &&  /bin/bash prepare.sh && cd -
cd rodinia && /bin/bash prepare.sh && cd -
cd simple && /bin/bash prepare.sh && cd -
cd annotated && /bin/bash prepare.sh && cd -