import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import argparse
import os
import sys
import shutil
from subprocess import call
from DeOldify.deoldify.visualize import *
import DeOldify.fastai
import torch 
from DeOldify.deoldify import device
from DeOldify.deoldify.device_id import DeviceId

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def modify(image_filename=None, cv2_frame=None):

    def run_cmd(command):
        try:
            call(command, shell=True)
        except KeyboardInterrupt:
            print("Process interrupted")
            sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        default=image_filename, help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Restored images, please use the absolute path",
    )
    parser.add_argument("--GPU", type=str, default="-1", help="0,1,2")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint"
    )
    parser.add_argument("--with_scratch", default="--with_scratch", action="store_true")
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # Resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    # Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    os.chdir("./Global")
    stage_1_input_dir = opts.input_folder
    fn = os.path.basename(stage_1_input_dir)
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")

    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    if not opts.with_scratch:
        stage_1_command = (
            f'python test.py --test_mode Full --Quality_restore --test_input "{stage_1_input_dir}" '
            f'--outputs_dir "{stage_1_output_dir}" --gpu_ids "{gpu1}"'
        )
        run_cmd(stage_1_command)
    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")

        stage_1_command_1 = (
            f'python detection.py --test_path "{stage_1_input_dir}" --output_dir "{mask_dir}" '
            f'--input_size full_size --GPU "{gpu1}"'
        )
        stage_1_command_2 = (
            f'python test.py --Scratch_and_Quality_restore --test_input "{stage_1_input_dir}" '
            f'--test_mask "{new_mask}" --outputs_dir "{stage_1_output_dir}" --gpu_ids "{gpu1}"'
        )
        run_cmd(stage_1_command_1)
        run_cmd(stage_1_command_2)

    # Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("Completed Stage 1 ...")
    print("\n")

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

    # Stage 5: Colorizing
    print("Running Stage 5: Colorizing")
    os.chdir(".././DeOldify")
    fn = os.path.splitext(fn)[0]
    stage_5_input_dir = os.path.join(stage_4_output_dir, fn + ".png")
    device.set(device=DeviceId.GPU0)
    torch.backends.cudnn.benchmark = True
    colorizer = get_image_colorizer(artistic=False)
    colorizer.plot_transformed_image(stage_5_input_dir, render_factor=10)

    print("All the processing is done. Please check the results.")


# --------------------------------- The GUI ---------------------------------

# First the window layout...
sg.theme('DarkGrey')
images_col = [
    [sg.Text('Input photo:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
    [sg.Button('Modify Photo', key='-MPHOTO-'), sg.Button('Exit')],
    [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],
]
# ----- Full layout -----
layout = [[sg.VSeperator(), sg.Column(images_col)]]

# ----- Make the window -----
window = sg.Window('Old photos restoration', layout, grab_anywhere=True)

# ----- Run the Event Loop -----
prev_filename = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break

    elif event == '-MPHOTO-':
        try:
            n1 = filename.split("/")[-2]
            n2 = filename.split("/")[-3]
            n3 = filename.split("/")[-1]
            n3 = n3.split(".")[0] + ".png"

            modify(filename)

            image = cv2.imread(f'D:/Bringing-Old-Photos-Back-to-Life-master/DeOldify/result_images/{n3}')
            window['-OUT-'].update(data=cv2.imencode('.png', image)[1].tobytes())
        except Exception as e:
            print(f"Error: {e}")
            continue

    elif event == '-IN FILE-':  # A single filename was chosen
        filename = values['-IN FILE-']
        if filename != prev_filename:
            prev_filename = filename
            try:
                image = cv2.imread(filename)
                window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            except Exception as e:
                print(f"Error: {e}")
                continue

# ----- Exit program -----
window.close()
