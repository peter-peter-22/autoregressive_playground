sudo apt update
cd to_instance
python3 -m venv venv

source venv/bin/activate

pip install torch gdown numpy tokenizers python-dotenv psutil
sudo ldconfig
python train.py

zip -r output.zip output/