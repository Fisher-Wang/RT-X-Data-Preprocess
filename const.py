## Retrived from https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g, on Aug 20th, 2024
DATASETS = [
    "fractal20220817_data",
    "kuka",
    "bridge",
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "nyu_door_opening_surprising_effectiveness",
    "viola",
    "berkeley_autolab_ur5",
    "toto",
    "language_table",
    "columbia_cairlab_pusht_real",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "maniskill_dataset_converted_externally_to_rlds",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "usc_cloth_sim_converted_externally_to_rlds",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_saytap_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "robo_net",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "asu_table_top_converted_externally_to_rlds",
    "stanford_robocook_converted_externally_to_rlds",
    "eth_agent_affordances",
    "imperialcollege_sawyer_wrist_cam",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "qut_dexterous_manpulation",
    "uiuc_d3field",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "cmu_playing_with_food",
    "cmu_play_fusion",
    "cmu_stretch",
    "berkeley_gnm_recon",
    "berkeley_gnm_cory_hall",
    "berkeley_gnm_sac_son",
    "robot_vqa",
    "droid",
    "conq_hose_manipulation",
    "dobbe",
    "fmb",
    "io_ai_tech",
    "mimic_play",
    "aloha_mobile",
    "robo_set",
    "tidybot",
    "vima_converted_externally_to_rlds",
    "spoc",
    "plex_robosuite",
]
## To get the version of the dataset, use the following script
# keys=$(yq -r 'to_entries | map(select(.value==null)) | from_entries | keys | .[]' obs_keys.yaml)
# echo $keys | while read key; do echo $key; gsutil ls -l gs://gresearch/robotics/$key; done
VERSIONS = {
    "aloha_mobile": "0.0.1",
    "cmu_playing_with_food": "1.0.0",
    "conq_hose_manipulation": "0.0.1",
    "dobbe": "0.0.1",
    "droid": "1.0.0",
    "fmb": "0.0.1",
    "io_ai_tech": "0.0.1",
    "mimic_play": "0.0.1",
    "plex_robosuite": "0.0.1",
    "robo_set": "0.0.1",
    "spoc": "0.0.1",
    "tidybot": "0.0.1",
    "vima_converted_externally_to_rlds": "0.0.1",
    "robo_net": "1.0.0",
    "language_table": "0.0.1",
}
