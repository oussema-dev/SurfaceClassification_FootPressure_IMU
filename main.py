import argparse
import pandas as pd
import os
from preprocessing.preprocess import (check_synchronization, transform_orientation, segment_gaits, 
trim_data, merge_data, update_segments, interpolate_data, calculate_features, combine_stat_features)
from constants import DATA_SET_FOLDER, ML_DATA, PREFIXES, DL_DATA, GROUPING_COL, DATA_FILES
from utils.feature_reduction import reduce_features
from models.xgboost import train_ml_model
from models.cnn import train_dl_model
from utils.summarize_metrics import print_metrics

def preprocess_data():
    check_synchronization()
    transform_orientation()
    segment_gaits()
    trim_data()
    merge_data()
    update_segments()
    interpolate_data()
    calculate_features()
    combine_stat_features()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ML/DL model with config and segmentation options.")
    
    parser.add_argument(
        "--model", 
        choices=["ml", "dl"], 
        required=True,
        help="Type of model to use: 'ml' or 'dl'"
    )
    
    parser.add_argument(
        "--config", 
        choices=["1", "2", "3", "4"], 
        required=True,
        help=(
            "Configuration options:\n"
            "1 = IMU_lower_limbs\n"
            "2 = IMU_luo\n"
            "3 = Pressure_insoles\n"
            "4 = Pressure_insoles + IMU_lower_limbs"
        )
    )
    
    parser.add_argument(
        "--segmentation", 
        choices=["gait", "sw"], 
        required=True,
        help="Segmentation method: 'gait segmentation' or 'sliding window'"
    )

    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="If specified, the data will be preprocessed from scratch, otherwise, the existing preprocessed data will be used"
    )
    
    return parser.parse_args()

def check_files_and_preprocess(preprocess=None): 
    all_files_present = all(os.path.exists(os.path.join(DATA_SET_FOLDER, f)) for f in DATA_FILES)
    if  preprocess or all_files_present == False:
        print(f"Preprocessing data...")
        preprocess_data()
        print(f"Preprocessing finished successfully")

def main():
    args = parse_arguments()
    check_files_and_preprocess(args.preprocess)
    if args.model == "ml":
        data_file = os.path.join(DATA_SET_FOLDER, ML_DATA[args.segmentation])
        df = pd.read_csv(data_file)
        df = df.dropna()
        cols_to_keep = [col for col in df.columns if any(col.startswith(prefix) for prefix in PREFIXES[args.config])]
        df = df[cols_to_keep]
        unique_vals = df["walk_mode"].unique()
        mapping_dict = {val: idx for idx, val in enumerate(unique_vals)}
        df["walk_mode"] = df["walk_mode"].map(mapping_dict)
        features = df.columns[:len(df.columns)-2].tolist()
        features = reduce_features(df, features, variance_threshold=0.1, max_corr=0.8)
        accuracies, f1_scores, sensitivities, specificities = train_ml_model(df, features, unique_vals)

    elif args.model == "dl":
        root_dir = DATA_SET_FOLDER
        dataset = DL_DATA[args.segmentation]
        grouping_col = GROUPING_COL[args.segmentation]
        cols = PREFIXES[args.config]
        cols.append(grouping_col)
        df = pd.read_csv(os.path.join(root_dir, dataset))
        df = df.dropna()
        df = df[cols]
        unique_vals = df["walk_mode"].unique()
        accuracies, f1_scores, sensitivities, specificities = train_dl_model(df, grouping_col, unique_vals)

    print_metrics(accuracies, f1_scores, sensitivities, specificities)
    
if __name__ == "__main__":
    main()
