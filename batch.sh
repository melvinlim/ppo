#!/bin/bash
#bash always does stuff sequentially hopefully
#for i in {1..3}; do
game="Tetris-GameBoy"
game="Pong-Atari2600"
path="recordings/$game"
echo $path
rm recordings/target
for i in $(ls $path); do
  echo "processing $path/$i."
  cp $path/$i recordings/target
  echo "file copied, starting python script."
  python ppo.py --game $game
  echo "script finished."
done
