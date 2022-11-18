# AVSR 
```bash
git clone https://github.com/SRI-CSL/TrinityMultimodalTrojAI.git

pip install fvcore=0.1.5
pip install catalogue-1.0.0 en-vectors-web-lg-2.1.0 plac-1.1.3 spacy-2.3.7 srsly-1.0.5 thinc-7.4.5

pip install spacy==2.1.0        
pip install spacy-legacy==3.0.10
pip install spacy-loggers==1.0.3
pip install en-vectors-web-lg==2.1.0
pip install torch==1.10.2
pip install pycocotools==2.0.5

cd  datagen/detectron2
pip install -e .
cd ../.. 

cd /media/xingshuo/home1/multimodal/TrinityMultimodalTrojAI/datagen
python optimize_patch.py --over 1>optimize_patch_log.txt 2>&1  &
```