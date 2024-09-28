#!/bin/bash


echo "SWEEP LR a 768"

python test.py --load_dir "runs/super-sun-141/" --batch_size 32
python test.py --load_dir "runs/rose-terrain-142/" --batch_size 32
python test.py --load_dir "runs/genial-dust-143/" --batch_size 32
python test.py --load_dir "runs/faithful-water-144/" --batch_size 32
python test.py --load_dir "runs/jumping-bee-145/" --batch_size 32
python test.py --load_dir "runs/gallant-snow-146/" --batch_size 32


echo "SWEEP LR muP a 64"

python test.py --load_dir "runs/good-dawn-147/" --batch_size 32
python test.py --load_dir "runs/ethereal-plasma-148/" --batch_size 32
python test.py --load_dir "runs/treasured-surf-149/" --batch_size 32
python test.py --load_dir "runs/likely-pond-150/" --batch_size 32
python test.py --load_dir "runs/brisk-brook-151/" --batch_size 32
python test.py --load_dir "runs/deft-music-152/" --batch_size 32
python test.py --load_dir "runs/astral-sun-153/" --batch_size 32


echo "SWEEP LR SP a 64"

python test.py --load_dir "runs/silver-sound-154/" --batch_size 32
python test.py --load_dir "runs/balmy-terrain-155/" --batch_size 32
python test.py --load_dir "runs/dauntless-puddle-156/" --batch_size 32
python test.py --load_dir "runs/driven-vally-157/" --batch_size 32
python test.py --load_dir "runs/comfy-forest-158/" --batch_size 32
python test.py --load_dir "runs/mild-frost-159/" --batch_size 32
python test.py --load_dir "runs/sleek-leaf-160/" --batch_size 32
python test.py --load_dir "runs/solar-haze-161/" --batch_size 32
