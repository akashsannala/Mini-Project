  # Stage 2: Face Detection
    print("Running Stage 2: Face Detection")
    os.chdir(".././Face_Detection")
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)
    stage_2_command = (
        f'python detect_all_dlib.py --url "{stage_2_input_dir}" --fn "{fn}" --save_url "{stage_2_output_dir}"'
    )
    run_cmd(stage_2_command)
    print("Completed Stage 2 ...")
    print("\n")