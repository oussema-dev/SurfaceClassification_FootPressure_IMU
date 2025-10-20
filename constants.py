DATA_SET_FOLDER='data_set'

SEGMENTS = [
        'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
        'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
        'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
        'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe',
        'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToe'
]

COLUMNS_MERGED = [
    'Left_Hallux_raw', 'Right_Hallux_raw',
    'Left_Toes_raw', 'Right_Toes_raw',
    'Left_Met1_raw', 'Left_Met3_raw', 'Left_Met5_raw',
    'Right_Met1_raw', 'Right_Met3_raw', 'Right_Met5_raw', 
    'Left_Arch_raw', 'Right_Arch_raw',
    'Left_Heel_R_raw', 'Left_Heel_L_raw', 
    'Right_Heel_L_raw', 'Right_Heel_R_raw',

    "acceleration_Pelvis_x_local","acceleration_Pelvis_y_local","acceleration_Pelvis_z_local",
    "acceleration_RightForeArm_x_local","acceleration_RightForeArm_y_local","acceleration_RightForeArm_z_local",
    "acceleration_RightUpperLeg_x_local","acceleration_RightUpperLeg_y_local","acceleration_RightUpperLeg_z_local",
    "acceleration_RightLowerLeg_x_local","acceleration_RightLowerLeg_y_local","acceleration_RightLowerLeg_z_local",
    "acceleration_RightFoot_x_local","acceleration_RightFoot_y_local","acceleration_RightFoot_z_local",
    "acceleration_RightToe_x_local","acceleration_RightToe_y_local","acceleration_RightToe_z_local",
    "acceleration_LeftUpperLeg_x_local","acceleration_LeftUpperLeg_y_local","acceleration_LeftUpperLeg_z_local",
    "acceleration_LeftLowerLeg_x_local","acceleration_LeftLowerLeg_y_local","acceleration_LeftLowerLeg_z_local",
    "acceleration_LeftFoot_x_local","acceleration_LeftFoot_y_local","acceleration_LeftFoot_z_local",
    "acceleration_LeftToe_x_local","acceleration_LeftToe_y_local","acceleration_LeftToe_z_local",

    "angularVelocity_Pelvis_x_local","angularVelocity_Pelvis_y_local","angularVelocity_Pelvis_z_local",
    "angularVelocity_RightForeArm_x_local","angularVelocity_RightForeArm_y_local","angularVelocity_RightForeArm_z_local",
    "angularVelocity_RightUpperLeg_x_local","angularVelocity_RightUpperLeg_y_local","angularVelocity_RightUpperLeg_z_local",
    "angularVelocity_RightLowerLeg_x_local","angularVelocity_RightLowerLeg_y_local","angularVelocity_RightLowerLeg_z_local",
    "angularVelocity_RightFoot_x_local","angularVelocity_RightFoot_y_local","angularVelocity_RightFoot_z_local",
    "angularVelocity_RightToe_x_local","angularVelocity_RightToe_y_local","angularVelocity_RightToe_z_local",
    "angularVelocity_LeftUpperLeg_x_local","angularVelocity_LeftUpperLeg_y_local","angularVelocity_LeftUpperLeg_z_local",
    "angularVelocity_LeftLowerLeg_x_local","angularVelocity_LeftLowerLeg_y_local","angularVelocity_LeftLowerLeg_z_local",
    "angularVelocity_LeftFoot_x_local","angularVelocity_LeftFoot_y_local","angularVelocity_LeftFoot_z_local",
    "angularVelocity_LeftToe_x_local","angularVelocity_LeftToe_y_local","angularVelocity_LeftToe_z_local",

    'participant_id',  'walk_mode'
]

WINDOW_SIZE = 120
STEP_SIZE = 60     

METADATA_COLS = ['participant_id', 'walk_mode', 'stepcount']

ML_DATA = {"gait": "combined_stat_freq_features(gait cycles).csv", "sw": "combined_stat_freq_features(sliding window).csv"}

PREFIXES = {
    "1": [
            "acceleration_Pelvis_x_local","acceleration_Pelvis_y_local","acceleration_Pelvis_z_local",
            "acceleration_RightUpperLeg_x_local","acceleration_RightUpperLeg_y_local","acceleration_RightUpperLeg_z_local",
            "acceleration_RightLowerLeg_x_local","acceleration_RightLowerLeg_y_local","acceleration_RightLowerLeg_z_local",
            "acceleration_RightFoot_x_local","acceleration_RightFoot_y_local","acceleration_RightFoot_z_local",
            "acceleration_RightToe_x_local","acceleration_RightToe_y_local","acceleration_RightToe_z_local",
            "acceleration_LeftUpperLeg_x_local","acceleration_LeftUpperLeg_y_local","acceleration_LeftUpperLeg_z_local",
            "acceleration_LeftLowerLeg_x_local","acceleration_LeftLowerLeg_y_local","acceleration_LeftLowerLeg_z_local",
            "acceleration_LeftFoot_x_local","acceleration_LeftFoot_y_local","acceleration_LeftFoot_z_local",
            "acceleration_LeftToe_x_local","acceleration_LeftToe_y_local","acceleration_LeftToe_z_local",

            "angularVelocity_Pelvis_x_local","angularVelocity_Pelvis_y_local","angularVelocity_Pelvis_z_local",
            "angularVelocity_RightUpperLeg_x_local","angularVelocity_RightUpperLeg_y_local","angularVelocity_RightUpperLeg_z_local",
            "angularVelocity_RightLowerLeg_x_local","angularVelocity_RightLowerLeg_y_local","angularVelocity_RightLowerLeg_z_local",
            "angularVelocity_RightFoot_x_local","angularVelocity_RightFoot_y_local","angularVelocity_RightFoot_z_local",
            "angularVelocity_RightToe_x_local","angularVelocity_RightToe_y_local","angularVelocity_RightToe_z_local",
            "angularVelocity_LeftUpperLeg_x_local","angularVelocity_LeftUpperLeg_y_local","angularVelocity_LeftUpperLeg_z_local",
            "angularVelocity_LeftLowerLeg_x_local","angularVelocity_LeftLowerLeg_y_local","angularVelocity_LeftLowerLeg_z_local",
            "angularVelocity_LeftFoot_x_local","angularVelocity_LeftFoot_y_local","angularVelocity_LeftFoot_z_local",
            "angularVelocity_LeftToe_x_local","angularVelocity_LeftToe_y_local","angularVelocity_LeftToe_z_local",

            'participant_id',  'walk_mode'
        ],
    "2": [
            "acceleration_Pelvis_x_local","acceleration_Pelvis_y_local","acceleration_Pelvis_z_local",
            "acceleration_RightForeArm_x_local","acceleration_RightForeArm_y_local","acceleration_RightForeArm_z_local",
            "acceleration_RightUpperLeg_x_local","acceleration_RightUpperLeg_y_local","acceleration_RightUpperLeg_z_local",
            "acceleration_RightLowerLeg_x_local","acceleration_RightLowerLeg_y_local","acceleration_RightLowerLeg_z_local",
            "acceleration_LeftUpperLeg_x_local","acceleration_LeftUpperLeg_y_local","acceleration_LeftUpperLeg_z_local",
            "acceleration_LeftLowerLeg_x_local","acceleration_LeftLowerLeg_y_local","acceleration_LeftLowerLeg_z_local",

            "angularVelocity_Pelvis_x_local","angularVelocity_Pelvis_y_local","angularVelocity_Pelvis_z_local",
            "angularVelocity_RightForeArm_x_local","angularVelocity_RightForeArm_y_local","angularVelocity_RightForeArm_z_local",
            "angularVelocity_RightUpperLeg_x_local","angularVelocity_RightUpperLeg_y_local","angularVelocity_RightUpperLeg_z_local",
            "angularVelocity_RightLowerLeg_x_local","angularVelocity_RightLowerLeg_y_local","angularVelocity_RightLowerLeg_z_local",
            "angularVelocity_LeftUpperLeg_x_local","angularVelocity_LeftUpperLeg_y_local","angularVelocity_LeftUpperLeg_z_local",
            "angularVelocity_LeftLowerLeg_x_local","angularVelocity_LeftLowerLeg_y_local","angularVelocity_LeftLowerLeg_z_local",

            'participant_id',  'walk_mode'
        ],
    "3": [
            'Left_Hallux_raw', 'Right_Hallux_raw',
            'Left_Toes_raw', 'Right_Toes_raw',
            'Left_Met1_raw', 'Left_Met3_raw', 'Left_Met5_raw',
            'Right_Met1_raw', 'Right_Met3_raw', 'Right_Met5_raw', 
            'Left_Arch_raw', 'Right_Arch_raw',
            'Left_Heel_R_raw', 'Left_Heel_L_raw', 
            'Right_Heel_L_raw', 'Right_Heel_R_raw',
            'participant_id',  'walk_mode'
        ],
    "4": [
            'Left_Hallux_raw', 'Right_Hallux_raw',
            'Left_Toes_raw', 'Right_Toes_raw',
            'Left_Met1_raw', 'Left_Met3_raw', 'Left_Met5_raw',
            'Right_Met1_raw', 'Right_Met3_raw', 'Right_Met5_raw', 
            'Left_Arch_raw', 'Right_Arch_raw',
            'Left_Heel_R_raw', 'Left_Heel_L_raw', 
            'Right_Heel_L_raw', 'Right_Heel_R_raw',

            "acceleration_Pelvis_x_local","acceleration_Pelvis_y_local","acceleration_Pelvis_z_local",
            "acceleration_RightUpperLeg_x_local","acceleration_RightUpperLeg_y_local","acceleration_RightUpperLeg_z_local",
            "acceleration_RightLowerLeg_x_local","acceleration_RightLowerLeg_y_local","acceleration_RightLowerLeg_z_local",
            "acceleration_RightFoot_x_local","acceleration_RightFoot_y_local","acceleration_RightFoot_z_local",
            "acceleration_RightToe_x_local","acceleration_RightToe_y_local","acceleration_RightToe_z_local",
            "acceleration_LeftUpperLeg_x_local","acceleration_LeftUpperLeg_y_local","acceleration_LeftUpperLeg_z_local",
            "acceleration_LeftLowerLeg_x_local","acceleration_LeftLowerLeg_y_local","acceleration_LeftLowerLeg_z_local",
            "acceleration_LeftFoot_x_local","acceleration_LeftFoot_y_local","acceleration_LeftFoot_z_local",
            "acceleration_LeftToe_x_local","acceleration_LeftToe_y_local","acceleration_LeftToe_z_local",

            "angularVelocity_Pelvis_x_local","angularVelocity_Pelvis_y_local","angularVelocity_Pelvis_z_local",
            "angularVelocity_RightUpperLeg_x_local","angularVelocity_RightUpperLeg_y_local","angularVelocity_RightUpperLeg_z_local",
            "angularVelocity_RightLowerLeg_x_local","angularVelocity_RightLowerLeg_y_local","angularVelocity_RightLowerLeg_z_local",
            "angularVelocity_RightFoot_x_local","angularVelocity_RightFoot_y_local","angularVelocity_RightFoot_z_local",
            "angularVelocity_RightToe_x_local","angularVelocity_RightToe_y_local","angularVelocity_RightToe_z_local",
            "angularVelocity_LeftUpperLeg_x_local","angularVelocity_LeftUpperLeg_y_local","angularVelocity_LeftUpperLeg_z_local",
            "angularVelocity_LeftLowerLeg_x_local","angularVelocity_LeftLowerLeg_y_local","angularVelocity_LeftLowerLeg_z_local",
            "angularVelocity_LeftFoot_x_local","angularVelocity_LeftFoot_y_local","angularVelocity_LeftFoot_z_local",
            "angularVelocity_LeftToe_x_local","angularVelocity_LeftToe_y_local","angularVelocity_LeftToe_z_local",

            'participant_id',  'walk_mode'
        ]

}

DL_DATA = {"gait": "merged_gait_segmentation_interpolated.csv", "sw": "merged_sliding_window.csv"}
GROUPING_COL = {"gait": "stepcount", "sw": "window_number"}

DATA_FILES = [
    "combined_stat_freq_features(gait cycles).csv", 
    "combined_stat_freq_features(sliding window).csv", 
    "merged_gait_segmentation_interpolated.csv",
    "merged_sliding_window.csv"
]