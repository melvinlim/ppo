#!/bin/bash
#bash always does stuff sequentially hopefully
for i in {1..3}; do
  echo "processing record$i"
  cp recordings/record$i recordings/target
  echo "file copied, starting python script."
  python ppo.py
  echo "script finished."
done
