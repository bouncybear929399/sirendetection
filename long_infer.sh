python long_infer.py "D:\警笛声检测测试集\real_wav" `
    --models-dir "D:\siren_models"`
    --device cpu --workers 4 --batch-size 16 `
    --time-smooth --high-thresh 0.90 `
    --window-size 3 --vote-needed 2 --min-conf 0.5 `
    --save-probs "probs_dump.csv" `
    --include-prob --out "mismatch_list_long.txt"