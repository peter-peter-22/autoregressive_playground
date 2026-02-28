sudo apt update
cd to_instance || exit
python3 -m venv venv

source venv/bin/activate

pip install torch gdown numpy tokenizers python-dotenv psutil async-mega-py==2.0.2 rich asyncio dotenv
sudo ldconfig
python train.py