# CAP-6411-Project

## Setup

```
conda create --name nuscenes python=3.9
conda activate nuscenes
pip install pyyaml
In file datasets/nuscenes_qa.py Replace en_vectors_web_lg -> en_core_web_lg
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install "numpy<2"
pip install ipdb
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz
pip install en_core_web_lg-2.1.0.tar.gz
python3 run.py --RUN='train' --MODEL='mcan_small' --VIS_FEAT='CenterPoint' --GPU='0, 1'

Download mcan_small.yml from https://github.com/MILVLG/openvqa/blob/master/configs/vqa/mcan_small.yml and save as configs/mcan_small.yaml
python -m spacy download en_core_web_lg
```