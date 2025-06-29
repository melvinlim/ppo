#!/bin/bash
#bash always does stuff sequentially hopefully
#for i in {1..3}; do
rm recordings/target
for i in $(ls recordings); do
  echo "processing $i."
  cp recordings/$i recordings/target
  echo "file copied, starting python script."
  python ppo.py
  echo "script finished."
done
