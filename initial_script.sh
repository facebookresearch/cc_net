#!/bin/bash

set -e

if [[ -d "/root/lm_sp" ]]; then
  echo "Folder /root/lm_sp already exists."
else
  echo "Folder /root/lm_sp does not exist."
  mkdir /root/lm_sp
fi

cp -n /dbfs/data-mle/llm/cc_net/lm_sp/* /root/lm_sp/

echo "Done copy models"