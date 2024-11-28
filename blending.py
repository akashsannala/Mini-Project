 # Stage 4: Warp back
    print("Running Stage 4: Blending")
    os.chdir(".././Face_Detection")
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    stage_4_command = (
        f'python align_warp_back_multiple_dlib.py --origin_url "{stage_4_input_image_dir}" '
        f'--fn "{fn}" --replace_url "{stage_4_input_face_dir}" --save_url "{stage_4_output_dir}"'
    )
    run_cmd(stage_4_command)
    print("Completed Stage 4 ...")
    print("\n")
