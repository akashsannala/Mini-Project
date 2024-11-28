 # Stage 3: Face Restore
    print("Running Stage 3: Face Enhancement")
    os.chdir(".././Face_Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)
    stage_3_command = (
        f'python test_face.py --old_face_folder "{stage_3_input_face}" --old_face_label_folder "{stage_3_input_mask}" '
        f'--tensorboard_log --name "{opts.checkpoint_name}" --gpu_ids "{gpu1}" '
        f'--load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 '
        f'--results_dir "{stage_3_output_dir}" --no_parsing_map'
    )
    run_cmd(stage_3_command)
    print("Completed Stage 3 ...")
    print("\n")
