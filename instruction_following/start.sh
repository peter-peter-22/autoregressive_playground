sudo apt update
python3 -m venv venv

source venv/bin/activate

pip install torch gdown numpy tokenizers python-dotenv psutil async-mega-py==2.0.2 rich asyncio dotenv datasets millify
sudo ldconfig

python train.py
python instruction_training.py