#!/bin/bash

set -eux

export EXAMPLE_ARGS=$1
export CHAINERCV_DOWNLOAD_REPORT="OFF"

for dir in `ls examples`
do
  if [[ -f examples/${dir}/export.py ]]; then
    python examples/${dir}/export.py -T ${EXAMPLE_ARGS}
  fi
done
