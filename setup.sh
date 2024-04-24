git clone https://github.com/zhangchuanyin/weed-datasets.git
pip install livelossplot --quiet
pip install tqdm --quiet
pip install gdown --quiet

gdown https://drive.google.com/uc?id=1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj
mkdir deepweeds
mkdir deepweeds/images
mv images.zip deepweeds/images/
cd /deepweeds/images && unzip images.zip
cd /deepweeds && wget https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv 