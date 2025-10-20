import os
import pandas as pd
import numpy as np
from natsort import natsorted
from scipy import integrate
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.stats import entropy, skew, kurtosis
from constants import DATA_SET_FOLDER, SEGMENTS, COLUMNS_MERGED, WINDOW_SIZE, STEP_SIZE, METADATA_COLS

# Function to check if the three CSV files have the same number of rows (check if the sync is okay)
def check_csv_files(folder_path):
    print("Checking folder", folder_path)
    
    # Remove unwanted files if they exist
    for unwanted_file in ["video.mp4", "eyetracker.csv"]:
        unwanted_path = os.path.join(folder_path, unwanted_file)
        if os.path.exists(unwanted_path):
            try:
                os.remove(unwanted_path)
                # print(f"Deleted {unwanted_file}")
            except Exception as e:
                print(f"Failed to delete {unwanted_file}: {e}")

    try:
        insoles_df = pd.read_csv(os.path.join(folder_path, "insoles.csv")).reset_index(drop=True)
        labels_df = pd.read_csv(os.path.join(folder_path, "labels.csv")).reset_index(drop=True)
        xsens_df = pd.read_csv(os.path.join(folder_path, "xsens.csv")).reset_index(drop=True)

        insoles_df.sort_values(by="time", inplace=True, ignore_index=True)
        labels_df.sort_values(by="time", inplace=True, ignore_index=True)
        xsens_df.sort_values(by="time", inplace=True, ignore_index=True)

        rows_insoles = len(insoles_df)
        rows_labels = len(labels_df)
        rows_xsens = len(xsens_df)

        if rows_insoles == rows_labels and rows_insoles == rows_xsens:
            merged_df = pd.concat([insoles_df, labels_df, xsens_df], axis=1)
            merged_df = merged_df.loc[~merged_df['walk_mode'].isin(['pavement_down', 'pavement_up'])]
            merged_df = merged_df.dropna().reset_index(drop=True)
            merged_df = merged_df[sorted(merged_df.columns)]
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            merged_df.to_csv(os.path.join(folder_path, "merged.csv"), index=False, encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"Error reading CSV files in {folder_path}: {e}")
        return False
    

def global2local(gyro_global, acc_global, quats):
    """
    Rotate gyroscope and acceleration data from global frame to sensor's local frame.

    Parameters:
        gyro_global (Nx3 array): angular velocity in global frame
        acc_global (Nx3 array): linear acceleration in global frame
        quats (Nx4 array): quaternions representing sensor orientation (w, x, y, z)

    Returns:
        gyro_local (Nx3 array): angular velocity in local frame
        acc_local (Nx3 array): linear acceleration in local frame
    """
    # Convert quaternion from (w, x, y, z) to (x, y, z, w) for scipy
    quats_xyzw = quats[:, [1, 2, 3, 0]]

    # Create Rotation object
    r_global_to_local = R.from_quat(quats_xyzw).inv()

    # Rotate gyroscope and acceleration from global to local
    gyro_local = r_global_to_local.apply(gyro_global)
    acc_local = r_global_to_local.apply(acc_global)

    return gyro_local, acc_local

def correct_all_segments_orientation(folder_path):
    """
    Applies orientation correction to all sensor segments in the dataset.

    Parameters:
        folder_path (String): Data file path.
    """

    print("Working on", folder_path)   
    df = pd.read_csv(os.path.join(folder_path, "merged.csv")).reset_index(drop=True)
    df.sort_values(by="time", inplace=True, ignore_index=True)

    segments = SEGMENTS

    all_local_data = []

    for segment in segments:
        acc_cols = [f'acceleration_{segment}_x', f'acceleration_{segment}_y', f'acceleration_{segment}_z']
        gyro_cols = [f'angularVelocity_{segment}_x', f'angularVelocity_{segment}_y', f'angularVelocity_{segment}_z']
        quat_cols = [f'orientation_{segment}_q1', f'orientation_{segment}_qi', f'orientation_{segment}_qj', f'orientation_{segment}_qk']

        # Check if all required columns exist
        if all(col in df.columns for col in acc_cols + gyro_cols + quat_cols):
            acc_global = df[acc_cols].values
            gyro_global = df[gyro_cols].values
            quats = df[quat_cols].values

            gyro_local, acc_local = global2local(gyro_global, acc_global, quats)

            # Create DataFrames with appropriate column names
            gyro_local_df = pd.DataFrame(gyro_local, columns=[f'{col}_local' for col in gyro_cols], index=df.index)
            acc_local_df = pd.DataFrame(acc_local, columns=[f'{col}_local' for col in acc_cols], index=df.index)

            all_local_data.append(gyro_local_df)
            all_local_data.append(acc_local_df)
        else:
            print(f"Skipping segment {segment}: missing columns.")

    df_local = pd.concat([df] + all_local_data, axis=1)
    df_local = df_local[sorted(df_local.columns)]
    df_local = df_local.loc[:, ~df_local.columns.duplicated()]
    df_local.to_csv(os.path.join(folder_path, "merged.csv"), index=False, encoding="utf-8")

def add_gait_count(folder_path):
    print("Working on", folder_path)   
    df = pd.read_csv(os.path.join(folder_path, "merged.csv")).reset_index(drop=True)
    df.sort_values(by="time", inplace=True, ignore_index=True)

    step_count = 0
    step_counts = []

    # Loop through each row and update the step count
    for is_step in df['insoles_LeftFoot_is_step']:
        if is_step:  # Increment the step count at each TRUE
            step_count += 1
        step_counts.append(step_count)

    df['stepcount'] = step_counts

    # Group by 'step' and filter out groups with more than one unique 'walk_mode' label
    filtered_df = (
        df.groupby('stepcount', sort=True)
        .filter(lambda x: x['walk_mode'].nunique() == 1)
    )

    # Loop through all gait cycles and filter out the ones that have a length outside of the acceptable range
    filtered_df = (
        filtered_df.groupby('stepcount', sort=True)
        .filter(lambda x: 41 <= len(x) <= 85)
    )
    filtered_df.sort_values(by='time', inplace=True, ignore_index=True)
    # Reassign the `stepcount` column to be continuous and start from 1
    filtered_df['stepcount'] = pd.factorize(filtered_df['stepcount'])[0] + 1
    filtered_df = filtered_df[sorted(filtered_df.columns)]
    filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
    filtered_df.to_csv(os.path.join(folder_path, "merged_gait_count_annotations.csv"), index=False, encoding="utf-8")

dataset_path = DATA_SET_FOLDER

def check_synchronization():
    for course_folder in natsorted(os.listdir(dataset_path)):
        course_folder_path = os.path.join(dataset_path, course_folder)
        if os.path.isdir(course_folder_path):
            for subfolder in natsorted(os.listdir(course_folder_path)):
                subfolder_path = os.path.join(course_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    if not check_csv_files(subfolder_path):
                        print(f"Mismatch in number of rows in folder: {subfolder_path}")
    print("Dataset merged successfully")

def transform_orientation():
    print("Adding local sensor orientation...")
    for course_folder in natsorted(os.listdir(dataset_path)):
        course_folder_path = os.path.join(dataset_path, course_folder)
        if os.path.isdir(course_folder_path):
            for subfolder in natsorted(os.listdir(course_folder_path)):
                subfolder_path = os.path.join(course_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    correct_all_segments_orientation(subfolder_path)
    print("Local sensor orientation added successfully")

def segment_gaits():
    print("Adding gait count according to the provided annotations...")
    for course_folder in natsorted(os.listdir(dataset_path)):
        course_folder_path = os.path.join(dataset_path, course_folder)
        if os.path.isdir(course_folder_path):
            for subfolder in natsorted(os.listdir(course_folder_path)):
                subfolder_path = os.path.join(course_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    add_gait_count(subfolder_path)
    print("Gait count added successfully")

root_dir = DATA_SET_FOLDER
columns_annotations = COLUMNS_MERGED + ['stepcount']
# Sliding window parameters
window_size = WINDOW_SIZE  # 2 seconds @ 60Hz
step_size = STEP_SIZE     # 1 second

def sliding_window_segment(df):
    windows = []
    window_number = 1
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size].copy()
        if window['walk_mode'].nunique() == 1:
            window['window_number'] = window_number
            windows.append(window)
            window_number += 1
    return pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()

def trim_data():
    print("Trimming data set...")
    for course in natsorted(['courseA', 'courseB', 'courseC']):
        course_path = os.path.join(root_dir, course)
        for participant in natsorted(os.listdir(course_path)):
            participant_path = os.path.join(course_path, participant)
            print("Working on", participant_path)
            merged_path = os.path.join(participant_path, 'merged.csv')
            annot_path = os.path.join(participant_path, 'merged_gait_count_annotations.csv')

            if os.path.isfile(merged_path):
                df_merged = pd.read_csv(merged_path).reset_index(drop=True)
                df_merged.sort_values(by="time", inplace=True, ignore_index=True)
                df_merged = df_merged[COLUMNS_MERGED]
                df_merged = df_merged[sorted(df_merged.columns)]
                df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
                segmented = sliding_window_segment(df_merged)
                segmented.to_csv(os.path.join(participant_path, 'sliding_window.csv'), index=False, encoding="utf-8")

            if os.path.isfile(annot_path):
                df_annot = pd.read_csv(annot_path).reset_index(drop=True)
                df_annot.sort_values(by="time", inplace=True, ignore_index=True)
                df_annot = df_annot[columns_annotations]
                df_annot = df_annot[sorted(df_annot.columns)]
                df_annot = df_annot.loc[:, ~df_annot.columns.duplicated()]
                df_annot.to_csv(os.path.join(participant_path, 'gait_segmentation.csv'), index=False, encoding="utf-8")
    print("Data set trimmed successfully")

def merge_data():
    print("Merging data files...")
    gait_segmentation_dfs = []
    sliding_window_dfs = []

    for course in natsorted(['courseA', 'courseB', 'courseC']):
        course_path = os.path.join(root_dir, course)
        for participant in natsorted(os.listdir(course_path)):
            participant_path = os.path.join(course_path, participant)
            print("Merging", participant_path)
            gait_seg_path = os.path.join(participant_path, 'gait_segmentation.csv')
            sliding_window_path = os.path.join(participant_path, 'sliding_window.csv')

            if os.path.isfile(gait_seg_path):
                df_gait = pd.read_csv(gait_seg_path).reset_index(drop=True)
                df_gait = df_gait[sorted(df_gait.columns)]
                df_gait = df_gait.loc[:, ~df_gait.columns.duplicated()]
                gait_segmentation_dfs.append(df_gait)

            if os.path.isfile(sliding_window_path):
                df_slide = pd.read_csv(sliding_window_path).reset_index(drop=True)
                df_slide = df_slide[sorted(df_slide.columns)]
                df_slide = df_slide.loc[:, ~df_slide.columns.duplicated()]
                sliding_window_dfs.append(df_slide)

    if gait_segmentation_dfs:
        merged_gait = pd.concat(gait_segmentation_dfs, ignore_index=True)
        merged_gait.to_csv(os.path.join(root_dir, 'merged_gait_segmentation.csv'), index=False, encoding="utf-8")

    if sliding_window_dfs:
        merged_slide = pd.concat(sliding_window_dfs, ignore_index=True)
        merged_slide.to_csv(os.path.join(root_dir, 'merged_sliding_window.csv'), index=False, encoding="utf-8")

    print("Data files merged successfully")

def update_segments():
    df_sw = pd.read_csv(os.path.join(root_dir, "merged_sliding_window.csv")).reset_index(drop=True)

    new_window_ids = []
    group_id = 0
    last_id = None

    for current_id in df_sw['window_number']:
        if current_id != last_id:
            group_id += 1
        new_window_ids.append(group_id)
        last_id = current_id

    df_sw['window_number'] = new_window_ids
    df_sw = df_sw[sorted(df_sw.columns)]
    df_sw = df_sw.loc[:, ~df_sw.columns.duplicated()]
    df_sw.to_csv(os.path.join(root_dir, "merged_sliding_window.csv"), index=False, encoding="utf-8")
    print("Updated 'window_number' in merged_sliding_window.csv")

    df_gait = pd.read_csv(os.path.join(root_dir, "merged_gait_segmentation.csv")).reset_index(drop=True)

    new_stepcounts = []
    group_id = 0
    last_step = None

    for current_step in df_gait['stepcount']:
        if current_step != last_step:
            group_id += 1
        new_stepcounts.append(group_id)
        last_step = current_step

    df_gait['stepcount'] = new_stepcounts
    df_gait = df_gait[sorted(df_gait.columns)]
    df_gait = df_gait.loc[:, ~df_gait.columns.duplicated()]
    df_gait.to_csv(os.path.join(root_dir, "merged_gait_segmentation.csv"), index=False, encoding="utf-8")
    print("Updated 'stepcount' in merged_gait_segmentation.csv")

def interpolate_data():
    df = pd.read_csv(os.path.join(root_dir, "merged_gait_segmentation.csv")).reset_index(drop=True)
    sensor_cols = [col for col in df.columns if col not in METADATA_COLS]
    # Group by participant_id and stepcount
    grouped = df.groupby(['participant_id', 'stepcount'], sort=False)  # sort=False preserves order
    resampled_dfs = []
    for _, group in grouped:
        # Check uniform walk_mode
        unique_walk_modes = group['walk_mode'].unique()
        if len(unique_walk_modes) != 1:
            continue  # Disregard groups with varying walk_mode
        # Extract metadata row (same for all in group)
        metadata = group[METADATA_COLS].iloc[0:1]  # Take first row's metadata
        # Sensor data
        sensor_data = group[sensor_cols]
        n_rows = len(sensor_data)
        if n_rows == 0:
            continue
        x_old = np.arange(n_rows)
        x_new = np.linspace(x_old.min(), x_old.max(), 100)
        
        resampled_sensor = pd.DataFrame()
        for col in sensor_cols:
            y_old = sensor_data[col].values
            if n_rows == 1:
                # Constant fill for single point
                resampled_sensor[col] = np.full(100, y_old[0])
                continue
            # Choose interpolation kind (linear for determinism; can change to 'cubic' if needed)
            kind = 'linear'
            if n_rows < 100:
                # For extrapolation, use linear by default (scipy allows bounds outside)
                kind = 'linear'
            f = interp1d(x_old, y_old, kind=kind, fill_value='extrapolate')
            y_new = f(x_new)
            resampled_sensor[col] = y_new
        # Tile metadata to 100 rows
        metadata_tiled = pd.DataFrame(np.repeat(metadata.values, 100, axis=0), columns=METADATA_COLS)
        # Combine
        resampled_group = pd.concat([metadata_tiled, resampled_sensor], axis=1)
        resampled_dfs.append(resampled_group)
    if resampled_dfs:
        final_df = pd.concat(resampled_dfs, ignore_index=True)

        # Ensure consistent column order
        cols_order = METADATA_COLS + sensor_cols
        cols_order = [col for col in cols_order if col in final_df.columns]
        final_df = final_df[cols_order]
        final_df = final_df[sorted(final_df.columns)]
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        final_df.to_csv(os.path.join(root_dir, "merged_gait_segmentation_interpolated.csv"), index=False, encoding="utf-8")
        print("Interpolation complete. Saved to 'merged_gait_segmentation_interpolated.csv'")
    else:
        print("No valid groups to interpolate.")

# This method uses a sliding window of 2 seconds with a step of 1 second (excluding the segments that have more than one class)
def calculate_statistical_features_sliding_window(folder_path):
    print("Working on", folder_path)
    df = pd.read_csv(os.path.join(folder_path, "merged.csv")).reset_index(drop=True)
    df['_original_index'] = np.arange(len(df))
    df = df[COLUMNS_MERGED + ['_original_index']]
    if 'stepcount' in df.columns:
        df = df.drop(columns=['stepcount'])
    # Invariant columns in the dataset (not features)
    additional_columns = ['participant_id',  'walk_mode', '_original_index']
    
    count_any_na = df.isna().any(axis=1).sum()
    if count_any_na > 0:
        print(folder_path, "contains a NaN")
        df.fillna(0, inplace=True)

    final_features_df = pd.DataFrame()
    
    # Window and step sizes based on the refresh rate of 60Hz
    window_size = WINDOW_SIZE
    step_size = STEP_SIZE
    
    unique_classes = sorted(df["walk_mode"].unique())

    for class_label in unique_classes:
        class_df = df[df["walk_mode"] == class_label].sort_values('_original_index').reset_index(drop=True)
        num_rows = len(class_df)

        for start in range(0, num_rows - window_size + 1, step_size):
            window_df = class_df.iloc[start:start + window_size]

            # Check if the window contains only one unique class label
            if window_df["walk_mode"].nunique() > 1:
                continue
            window_features = {}

            try:
                for feature in df.columns:  
                    if feature not in additional_columns: # Skip non-feature columns
                        feature_values = window_df[feature]

                        # Statistical features
                        window_features[feature + "_mean"] = feature_values.mean()
                        window_features[feature + "_min"] = feature_values.min()
                        window_features[feature + "_max"] = feature_values.max()
                        window_features[feature + "_std"] = feature_values.std()
                        window_features[feature + "_iqr"] = np.percentile(feature_values, 75) - np.percentile(feature_values, 25)
                        window_features[feature + "_mad"] = np.median(np.abs(feature_values - np.median(feature_values)))
                        window_features[feature + "_auc"] = integrate.simpson(feature_values)

                        # Entropy (Histogram-based method)
                        hist, bin_edges = np.histogram(feature_values, bins=10, density=True)
                        hist = hist[hist > 0]  # Remove zero entries to avoid log(0) in entropy
                        window_features[feature + "_entropy"] = entropy(hist)

                        # Skewness and Kurtosis
                        window_features[feature + "_skewness"] = skew(feature_values)
                        window_features[feature + "_kurtosis"] = kurtosis(feature_values)

                        # Frequency domain features
                        # Calculate the FFT
                        fft_values = np.fft.fft(feature_values)
                        dft_magnitude = np.abs(fft_values)

                        # Store the first five DFT coefficients
                        window_features[feature + "_dft1"] = dft_magnitude[1]
                        window_features[feature + "_dft2"] = dft_magnitude[2]
                        window_features[feature + "_dft3"] = dft_magnitude[3]
                        window_features[feature + "_dft4"] = dft_magnitude[4]
                        window_features[feature + "_dft5"] = dft_magnitude[5]

                        # Weighted Mean Frequency Calculation
                        frequencies = np.fft.fftfreq(len(feature_values))
                        weighted_mean_frequency = np.sum(frequencies[1:6] * dft_magnitude[1:6]) / np.sum(dft_magnitude[1:6])
                        window_features[feature + "_weighted_mean_frequency"] = weighted_mean_frequency

                # Add additional columns to the window features
                for col in additional_columns:
                    window_features[col] = window_df[col].iloc[0]  # Assuming all values are the same in the window
                window_features["walk_mode"] = class_label
                window_features["window_start"] = start
                window_features["window_end"] = start + window_size

                final_features_df = pd.concat([final_features_df, pd.DataFrame([window_features])], ignore_index=True)

            except Exception as e:
                print(f"Error processing class {class_label} at window {start}-{start + window_size}: {e}")
    final_features_df = final_features_df.drop('_original_index', axis=1)
    final_features_df = final_features_df[sorted(final_features_df.columns)]
    final_features_df = final_features_df.loc[:, ~final_features_df.columns.duplicated()]
    final_features_df.to_csv(os.path.join(folder_path, "stat_freq_features(sliding window).csv"), index=False, encoding="utf-8")

# This method calculates statistical and frequency-based features for each batch having the same cycle
# It also excludes batches having different classes
def calculate_statistical_features_gait_cycles(folder_path):
    print("Working on", folder_path)
    df = pd.read_csv(os.path.join(folder_path, "merged_gait_count_annotations.csv"))
    # Keep original row order for deterministic processing
    df['_original_index'] = np.arange(len(df))

    # Keep only the required columns
    column_names = COLUMNS_MERGED + ['stepcount']
    df = df[column_names + ['_original_index']]

    # Metadata columns to keep per group
    additional_columns = ['participant_id', 'walk_mode', 'stepcount']

    # Fill NaNs deterministically
    if df.isna().any().any():
        print(folder_path, "contains NaNs, filling with 0")
        df.fillna(0, inplace=True)

    final_features_df = pd.DataFrame()

    # Sort stepcounts to ensure deterministic group processing
    sorted_stepcounts = sorted(df['stepcount'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)

    for stepcount in sorted_stepcounts:
        group = df[df['stepcount'] == stepcount].sort_values('_original_index').reset_index(drop=True)
        unique_walk_modes = group['walk_mode'].unique()

        # Skip groups with inconsistent walk_mode
        if len(unique_walk_modes) != 1:
            continue

        current_class = unique_walk_modes[0]
        segment_features = {}

        try:
            for feature in df.columns:
                if feature in additional_columns + ['_original_index']:
                    continue

                values = group[feature].astype(float).values

                # Statistical features
                segment_features[feature + "_mean"] = np.mean(values)
                segment_features[feature + "_min"] = np.min(values)
                segment_features[feature + "_max"] = np.max(values)
                segment_features[feature + "_std"] = np.std(values, ddof=0)
                segment_features[feature + "_iqr"] = np.percentile(values, 75) - np.percentile(values, 25)
                segment_features[feature + "_mad"] = np.median(np.abs(values - np.median(values)))
                segment_features[feature + "_auc"] = integrate.simpson(values)

                # Entropy
                hist, _ = np.histogram(values, bins=10, density=True)
                hist = hist[hist > 0]
                segment_features[feature + "_entropy"] = entropy(hist)

                # Skewness and kurtosis
                segment_features[feature + "_skewness"] = skew(values)
                segment_features[feature + "_kurtosis"] = kurtosis(values)

                # Frequency domain
                fft_values = np.fft.fft(values)
                dft_magnitude = np.abs(fft_values)
                for i in range(1, 6):
                    segment_features[f"{feature}_dft{i}"] = dft_magnitude[i]
                frequencies = np.fft.fftfreq(len(values))
                weighted_mean_frequency = np.sum(frequencies[1:6] * dft_magnitude[1:6]) / np.sum(dft_magnitude[1:6])
                segment_features[feature + "_weighted_mean_frequency"] = weighted_mean_frequency

            # Add metadata columns
            for col in additional_columns:
                segment_features[col] = group[col].iloc[0]

            segment_features["walk_mode"] = current_class

            final_features_df = pd.concat([final_features_df, pd.DataFrame([segment_features])], ignore_index=True)

        except Exception as e:
            print(f"Error processing stepcount {stepcount}: {e}")

    # Remove duplicate columns and sort alphabetically
    final_features_df = final_features_df.loc[:, ~final_features_df.columns.duplicated()]
    final_features_df = final_features_df[sorted(final_features_df.columns)]
    final_features_df.to_csv(os.path.join(folder_path, "stat_freq_features(gait cycles).csv"), index=False, encoding="utf-8")

def calculate_features():
    print("Calculating statistical features for the sliding window approach...")
    for course_folder in natsorted(os.listdir(dataset_path)):
        course_folder_path = os.path.join(dataset_path, course_folder)
        if os.path.isdir(course_folder_path):
            for subfolder in natsorted(os.listdir(course_folder_path)):
                subfolder_path = os.path.join(course_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    calculate_statistical_features_sliding_window(subfolder_path)
    print("Statistical features calculated successfully")

    print("Calculating statistical features for the gait segmentation approach...")
    for course_folder in natsorted(os.listdir(dataset_path)):
        course_folder_path = os.path.join(dataset_path, course_folder)
        if os.path.isdir(course_folder_path):
            for subfolder in natsorted(os.listdir(course_folder_path)):
                subfolder_path = os.path.join(course_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    calculate_statistical_features_gait_cycles(subfolder_path)
    print("Statistical features calculated successfully")

base_dir = DATA_SET_FOLDER

def merge_stat_features_sw():
    dfs = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = natsorted(dirs)  # sort subdirectories in-place for deterministic traversal
        files = natsorted(files)   # sort files deterministically

        if 'stat_freq_features(sliding window).csv' in files:
            csv_file_path = os.path.join(root, 'stat_freq_features(sliding window).csv')
            print("Merging", csv_file_path)
            df = pd.read_csv(csv_file_path, dtype=str, na_filter=True).reset_index(drop=True)
            dfs.append(df)

    if dfs:
        # Concatenate all DataFrames vertically
        combined_df = pd.concat(dfs, ignore_index=True)
        # Remove duplicate columns and sort alphabetically
        combined_df = combined_df[sorted(combined_df.columns)]
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        # Save CSV deterministically
        output_path = os.path.join(base_dir, 'combined_stat_freq_features(sliding window).csv')
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            combined_df.to_csv(f, index=False)
    print("Merging complete")

def merge_stat_features_gc():
    dfs = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = natsorted(dirs)  # deterministic order for subdirectories
        files = natsorted(files)   # deterministic file order

        if 'stat_freq_features(gait cycles).csv' in files:
            csv_file_path = os.path.join(root, 'stat_freq_features(gait cycles).csv')
            print("Merging", csv_file_path)
            df = pd.read_csv(csv_file_path, dtype=str, na_filter=True).reset_index(drop=True)
            dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df[sorted(combined_df.columns)]
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        output_path = os.path.join(base_dir, 'combined_stat_freq_features(gait cycles).csv')
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            combined_df.to_csv(f, index=False)
    print("Merging complete")

def combine_stat_features():
    print("Combining statistical and frequency features...")
    merge_stat_features_sw()
    merge_stat_features_gc()
    print("Statistical and frequency features combined successfully")