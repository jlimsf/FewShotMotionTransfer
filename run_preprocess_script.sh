# python data_preprocess/merge_background.py /data/top_data/test
# python data_preprocess/connect_body.py /data/top_data/test

# python data_preprocess/merge_background.py /data/top_data/train
# python data_preprocess/connect_body.py /data/top_data/train
python data_preprocess/UVToTexture.py /data/top_data/train


python data_preprocess/UVToTexture.py /data/top_data/test
