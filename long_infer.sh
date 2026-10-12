python long_infer_v1.py \
    "/mnt/e/BaiduNetdiskDownload/救护车警笛/processed/救护车处理后" \
    --models-dir "./model_16k" \
    --out-dir "./model_eval_results_20261009_v1_plots" \
    --device cpu \
    --workers 4 \
    --batch-size 16 \
    --time-smooth \
    --high-thresh 0.95 \
    --window-size 3 \
    --vote-needed 2 \
    --min-conf 0.60 \
    --save-probs "probs_dump.csv" \
    --save-windows \
    --save-switches \
    --plot-switches \
    --include-prob