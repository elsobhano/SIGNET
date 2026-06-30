# mT5 backbone (https://huggingface.co/google/mt5-base).
mt5_path = "./pretrained_weight/mt5-base"

# Annotation paths. Large-scale corpora (CSL_News, YT_ASL, BOBSL) use one JSON for all splits.
train_label_paths = {
                    "CSL_News": "./datasets/CSL_News/CSL_News_Labels.json",
                    "YT_ASL": "./datasets/YT_ASL/YT_ASL_Labels.json",
                    "BOBSL": "./datasets/BOBSL/BOBSL_Labels.json",
                    "CSL_Daily": "./datasets/CSL_Daily/labels.train",
                    "Phoenix": "./datasets/Phoenix/labels.train",
                    "How2Sign": "./datasets/How2Sign/how2sign_cleaned_train.csv",
                    "WLASL": "./datasets/WLASL/labels-2000.train",
                    "MeinDGS": "./datasets/MeinDGS/train.csv"
                    }

dev_label_paths = {
                    "CSL_News": "./datasets/CSL_News/CSL_News_Labels.json",
                    "YT_ASL": "./datasets/YT_ASL/YT_ASL_Labels.json",
                    "BOBSL": "./datasets/BOBSL/BOBSL_Labels.json",
                    "CSL_Daily": "./datasets/CSL_Daily/labels.dev",
                    "Phoenix": "./datasets/Phoenix/labels.dev",
                    "How2Sign": "./datasets/How2Sign/how2sign_cleaned_val.csv",
                    "WLASL": "./datasets/WLASL/labels-2000.dev",
                    "MeinDGS": "./datasets/MeinDGS/dev.csv"
                    }

test_label_paths = {
                    "CSL_News": "./datasets/CSL_News/CSL_News_Labels.json",
                    "YT_ASL": "./datasets/YT_ASL/YT_ASL_Labels.json",
                    "BOBSL": "./datasets/BOBSL/BOBSL_Labels.json",
                    "CSL_Daily": "./datasets/CSL_Daily/labels.test",
                    "Phoenix": "./datasets/Phoenix/labels.test",
                    "How2Sign": "./datasets/How2Sign/how2sign_cleaned_test.csv",
                    "WLASL": "./datasets/WLASL/labels-2000.test",
                    "MeinDGS": "./datasets/MeinDGS/test.csv"
                    }


# Pose (keypoint) dirs.
pose_dirs = {
            "CSL_News": "./dataset/CSL_News/pose_format",
            "YT_ASL": "./dataset/YT_ASL/pose_format",
            "BOBSL": "./dataset/BOBSL/pose_format",
            "CSL_Daily": "./dataset/CSL_Daily/pose_format",
            "Phoenix": "./dataset/Phoenix/pose_format",
            "How2Sign": "./dataset/How2Sign/pose_format",
            "WLASL": "./dataset/WLASL/pose_format",
            "MeinDGS": "./dataset/MeinDGS/pose_format"
            }

# W&B config. Set the API key via the WANDB_API_KEY env var, not here.
WANDB_CONFIG = {
            "WANDB_IGNORE_GLOBS": "*.patch",
            "WANDB_DISABLE_CODE": "true",
            "TOKENIZERS_PARALLELISM": "false",
        }
