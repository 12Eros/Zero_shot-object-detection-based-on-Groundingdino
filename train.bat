@echo off
cd /d F:\Open-GroundingDino-main\Open-GroundingDino-main
python main.py --config config/cfg_odvg_version2.py --datasets config/datasets_coco2017_splited_to_65_seen_odvg.json --output_dir ./logs --num_workers 8
pause