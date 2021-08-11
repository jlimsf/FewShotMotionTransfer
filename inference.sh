# python -W ignore finetune.py config/config_finetune.yaml \
#               --target /data/FSMR_data/top_data/train/91-kqBbzDIS \
#               --source sample_RT/sample1 \
#               --epochs 0 \
#               --device 0

# aws s3 cp video/ubc_checkpoints/output.mp4 s3://cqdatascience/john/

python -W ignore finetune.py config/config_finetune.yaml \
              --target /data/FSMR_data/top_data/train/91-kqBbzDIS \
              --source /data/FSMR_data/rebecca_taylor_top_v3PNG/test/000019B126/subject_1  \
              --epochs 0 \
              --device 0

aws s3 cp video/ubc_checkpoints/output.mp4 s3://cqdatascience/john/


# /data/FSMR_data/rebecca_taylor_top_v3PNG/train/016997D378/subject_1

# /data/FSMR_data/top_data/test/91cC+1+C4SS

# /data/FSMR_data/rebecca_taylor_top_v2/train/017019B048/subject_1
# sample_RT/sample1

# /data/FSMR_data/rebecca_taylor_top_v2/train/016997D378/subject_1
