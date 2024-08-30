import os
import tqdm

s3d_path = "~/Downloads/Structured3D"

# loop over the folder
for scene in tqdm.tqdm(sorted(os.listdir(s3d_path))):
    scene_id = int(scene.split("_")[-1])
    scene_dir = os.path.join(s3d_path, scene)

    if os.path.exists(os.path.join(scene_dir, "2D_rendering")):
        # otherwise it is already gathered
        imgs_path = []
        poses = []
        for traj in sorted(os.listdir(os.path.join(scene_dir, "2D_rendering"))):
            traj_dir = os.path.join(scene_dir, "2D_rendering", traj)

            for sub_traj in sorted(os.listdir(os.path.join(traj_dir, "perspective", "full"))):
                sub_traj_dir = os.path.join(traj_dir, "perspective", "full", sub_traj)

                # read img
                imgs_path.append(os.path.join(sub_traj_dir, "rgb_rawlight.png"))
                # read pose
                pose_file = os.path.join(sub_traj_dir, "camera_pose.txt")
                with open(pose_file, "r") as f:
                    pose_txt = [line.strip() for line in f.readlines()][0]
                poses.append(pose_txt)
        
        # make new directory for img
        new_img_dir = os.path.join(scene_dir, "imgs")
        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        # store new imgs
        for i, img_path in enumerate(imgs_path):
            os.system("cp " + img_path + " " + os.path.join(new_img_dir, str(i)+".png"))

        # save poses in poses.txt
        new_pose_file = os.path.join(scene_dir, "poses.txt")
        with open(new_pose_file, "w") as f:
            f.writelines([pose_txt+"\n" for pose_txt in poses])

        # remove old folder
        os.system("rm -rf "+os.path.join(scene_dir, "2D_rendering"))
