import os
import subprocess
import time
from pathlib import Path

from colorama import Fore

DATA_ROOT_DIR = Path("D:\\InstantSplat\\data").resolve()
dp_CodeRootDir = Path("D:\\InstantSplat\\").resolve()
dp_OutputModelDir = Path("D:\\InstantSplat\\output").resolve()
dp_OutputModelDir.mkdir(parents=True, exist_ok=True)
# print("Output Model Directory:", dp_OutputModelDir)
print(Fore.CYAN + "Output Model Directory:", dp_OutputModelDir)

DATASETS = [
    # "TT",
    # "sora",
    # "mars",
    "custom"
]

SCENES = [
    # "Family",
    # "Barn",
    # "Francis",
    # "Horse",
    # "Ignatius",
    # "santorini",
    # "parkingindoor",
    # "socket",
    # "socket-2",
    # "socket-1",
    # "socket-cad"
    # "fisheye",
    # "fisheyefront",
    # "fisheyeL+R+F"
    "zm-garage",
]


N_VIEWS = [
    # 3,
    # 5,
    # 7,
    # 9,
    # 10,
    # 12,
    # 15,
    # 20,
    30,
    # 40
]

# Increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter = 8000  # 2min30s c.a. on 8G RoG for 1000 iter. 1200 s, 20 min for 8000 iter, 30 min for 12000 iter.
pose_lr = "1x"

start_time = time.time()

messages = []

for DATASET in DATASETS:
    for SCENE in SCENES:
        for N_VIEW in N_VIEWS:
            current_time = time.time()
            # SOURCE_PATH must be Absolute path
            SOURCE_PATH = os.path.join(DATA_ROOT_DIR, DATASET, SCENE, f"{N_VIEW}_views")
            # MODEL_PATH = os.path.join(".", "output", "infer", DATASET, SCENE, f"{N_VIEW}_views_{gs_train_iter}Iter_{pose_lr}PoseLR")
            dp_ModelPath = dp_OutputModelDir / f"infer{gs_train_iter}" / DATASET / SCENE / f"{N_VIEW}_views_{gs_train_iter}Iter_{pose_lr}PoseLR"
            dn_ModelPath = dp_ModelPath.as_posix()
            # ----- (1) Dust3r_coarse_geometric_initialization -----
            fn_infer = (dp_CodeRootDir / "coarse_init_infer.py").as_posix()
            CMD_D1 = f"python -W ignore {fn_infer} --img_base_path {SOURCE_PATH} --n_views {N_VIEW} --focal_avg"

            # ----- (2) Train: jointly optimize pose -----
            CMD_T = f"python -W ignore ./train_joint.py -s {SOURCE_PATH} -m {dn_ModelPath} --n_views {N_VIEW} --scene {SCENE} --iter {gs_train_iter} --optim_pose"

            # ----- (3) Render interpolated pose & output video -----
            CMD_RI = f"python -W ignore ./render_by_interp.py -s {SOURCE_PATH} -m {dn_ModelPath} --n_views {N_VIEW} --scene {SCENE} --iter {gs_train_iter} --eval --get_video"

            try:
                print(f"========= {SCENE}: Dust3r_coarse_geometric_initialization =========")
                subprocess.run(CMD_D1, shell=True, check=True)
                print(f"========= {SCENE}: Train: jointly optimize pose =========")
                subprocess.run(CMD_T, shell=True, check=True)
                print(f"========= {SCENE}: Render interpolated pose & output video =========")
                subprocess.run(CMD_RI, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                messages.append(f"Error: {e}")
                print(Fore.RED + f"Error: {e}")
                continue

            time_message = f"Time taken for {SCENE} with {N_VIEW} views: {time.time() - current_time} seconds"
            messages.append(time_message)

print(Fore.GREEN + "Total time taken: ", time.time() - start_time)
print(Fore.GREEN + "All done!")
# if there are messages, print them
for message in messages:
    print(Fore.RED + message)
