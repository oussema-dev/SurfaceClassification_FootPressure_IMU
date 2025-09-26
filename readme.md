# Walking classification based on inertial measurement unit and foot pressure sensor data

### Note: the steps to run the code in this repo were performed on a windows platform 

- Download the dataset from https://springernature.figshare.com/collections/NEWBEE_A_Multi-Modal_Gait_Database_of_Natural_Everyday-Walk_in_an_Urban_Environment/5758997/1
- Extract the downloaded data and put the `data_set` folder in the project root directory

## Create conda environment

cd to the root folder and run the following commands:
- `conda create --name honda python=3.12.4 -y`
- `conda activate honda`
- `pip install -r requirements.txt`

## Run the project for the first time (data preprocessing)

- cd to the root folder
- type the command `python main.py --model <x> --config <y> --segmentation <z> --preprocess`

## Options

### `--model` (required)  
Select the type of model to use:  
- `ml` → Machine Learning–based model (XGBoost) 
- `dl` → Deep Learning–based model (1-D CNN)

### `--config` (required)  
Choose the dataset configuration:  
- `1` → IMU_lower_limbs  
- `2` → IMU_luo  
- `3` → Pressure_insoles  
- `4` → Pressure_insoles + IMU_lower_limbs  

### `--segmentation` (required)  
Specify the segmentation method:  
- `gait` → Gait segmentation  
- `sw` → Sliding window segmentation  

### `--preprocess` (optional flag but required for the first time)  
- If included, the data will be **preprocessed from scratch**.  
- If omitted, the script will use **existing preprocessed data** (if available). 

## Example Commands

- `python main.py --model ml --config 1 --segmentation gait --preprocess`
- `python main.py --model ml --config 2 --segmentation sw`