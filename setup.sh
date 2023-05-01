conda init zsh
conda create --name dlav_g21 python=3.10.9 -y
source activate dlav_g21
pip install cmake==3.26.3
pip install -r requirements.txt