#!/bin/bash

set -e

if [[ -d "/root/lm_sp" ]]; then
  echo "Folder /root/lm_sp already exists."
else
  echo "Folder /root/lm_sp does not exist."
  mkdir /root/lm_sp
fi

# cp -n /dbfs/data-mle/llm/cc_net/lm_sp/* /root/lm_sp/
cp -n /dbfs/data-mle/llm/cc_net/lm_sp/en.arpa.bin /root/lm_sp/
cp -n /dbfs/data-mle/llm/cc_net/lm_sp/en.sp.model /root/lm_sp/

pip install /dbfs/data-mle/llm/cc_net/dist/cc_net-1.0.0-py3-none-any.whl
pip install cc_net[getpy]

echo "Done copy models"