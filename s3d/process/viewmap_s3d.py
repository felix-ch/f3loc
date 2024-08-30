import numpy as np
import os, json
from s3d.process.s3d_utils import *
import matplotlib.pyplot as plt
import tqdm
import time
from scipy.spatial.transform import Rotation
from utils.utils import *


def angle_diff(a1, a2):
    da = np.abs((a1-a2))%(2*np.pi)
    return np.minimum(2*np.pi-da, da)

visualize = False
save = True
do_map = False
data_dir = "~/Downloads/Structured3D"

scenes = sorted(os.listdir(data_dir))[:]

# iteration over all scenes:
for scene in tqdm.tqdm(scenes):

    start_time = time.time()

    json_path = os.path.join(data_dir, scene, "annotation_3d.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    n_rooms, room_lines, door_lines, window_lines = read_s3d_floorplan(data)

    # do hough transformation for the both the room lines and the door lines line-> rho, theta
    # rho in R, theta [0, pi]
    room_x1 = room_lines[:,0,0]
    room_y1 = room_lines[:,0,1]
    room_x2 = room_lines[:,1,0]
    room_y2 = room_lines[:,1,1]
    room_thetas = np.arctan2(room_y1-room_y2, room_x1-room_x2) + np.pi/2
    room_rhos = room_x1*np.cos(room_thetas) + room_y1*np.sin(room_thetas)
    # make theta [-pi, pi], rho>=0
    room_thetas[room_rhos<0] = room_thetas[room_rhos<0]-np.pi
    room_rhos = np.abs(room_rhos)


    door_x1 = door_lines[:,0,0]
    door_y1 = door_lines[:,0,1]
    door_x2 = door_lines[:,1,0]
    door_y2 = door_lines[:,1,1]
    door_thetas = np.arctan2(door_y1-door_y2, door_x1-door_x2) + np.pi/2
    door_rhos = door_x1*np.cos(door_thetas) + door_y1*np.sin(door_thetas)
    # make theta [-pi, pi], rho>=0
    door_thetas[door_rhos<0] = door_thetas[door_rhos<0]-np.pi
    door_rhos = np.abs(door_rhos)

    # get proper size of the map
    if door_lines.shape[0] > 0:
        x_min = np.min([door_x1.min(), door_x2.min(), room_x1.min(), room_x2.min()])
        x_max = np.max([door_x1.max(), door_x2.max(), room_x1.max(), room_x2.max()])
        y_min = np.min([door_y1.min(), door_y2.min(), room_y1.min(), room_y2.min()])
        y_max = np.max([door_y1.max(), door_y2.max(), room_y1.max(), room_y2.max()])
    else:
        x_min = np.min([room_x1.min(), room_x2.min()])
        x_max = np.max([room_x1.max(), room_x2.max()])
        y_min = np.min([room_y1.min(), room_y2.min()])
        y_max = np.max([room_y1.max(), room_y2.max()])
    
    x_min = x_min // 0.5 * 0.5 #reserve 0.5 meter for boundary
    x_max = (x_max//0.5+1)*0.5
    y_min = y_min//0.5 * 0.5
    y_max = (y_max//0.5+1)*0.5


    if do_map:
        if visualize:
            fig = plt.figure("whole")
            fig.clf()
            plt.axis("equal")
            for line in room_lines:
                plt.plot(line[:,0], line[:,1], "b")

            for i, line in enumerate(door_lines):
                plt.plot(line[:,0], line[:,1], "r")
                plt.plot([0,np.cos(door_thetas[i])*door_rhos[i]], [0,np.sin(door_thetas[i])*door_rhos[i]], "g")
                plt.plot(door_x1[i], door_y1[i], "gx")
                plt.plot(door_x2[i], door_y2[i], "gx")

                # print(door_rhos[i])
                # plt.pause(0.01)

            # plt.imshow(room_map, cmap="gray", vmin=0, vmax=255)
            plt.pause(0.01)

        rho_threshold = 0.001
        theta_threshold = 0.002
        keep_door_line = np.ones_like(door_rhos, dtype=np.bool)

        # loop over door_rho
        for i, (door_rho, door_theta) in enumerate(zip(door_rhos, door_thetas)):
            if visualize:
                fig = plt.figure("whole")
                plt.plot([door_x1[i], door_x2[i]], [door_y1[i],door_y2[i]], 'r')
                plt.pause(0.01)
            # check if there is room_rho nearby
            rho_pass = np.abs(door_rho - room_rhos) < rho_threshold
            if np.any(rho_pass):
                # further check angles
                theta_pass = angle_diff(door_theta, room_thetas[rho_pass])<theta_threshold
                if any(theta_pass):
                    # there might be door lines overlapped with room lines
                    # check if the two ends of room line include the door line
                    #
                    passed_room_x1 = room_x1[rho_pass][theta_pass] #(Nrp)
                    passed_room_x2 = room_x2[rho_pass][theta_pass]
                    passed_room_y1 = room_y1[rho_pass][theta_pass]
                    passed_room_y2 = room_y2[rho_pass][theta_pass]

                    # calculate the projected coordinate
                    ortho_line = np.array([np.sin(door_theta), -np.cos(door_theta)]) #(2,)
                    projected_room_1 = passed_room_x1*ortho_line[0] + passed_room_y1*ortho_line[1] #(Nrp,)
                    projected_room_2 = passed_room_x2*ortho_line[0] + passed_room_y2*ortho_line[1] #(Nrp,)

                    projected_door_1 = door_x1[i]*ortho_line[0] + door_y1[i]*ortho_line[1] #()
                    projected_door_2 = door_x2[i]*ortho_line[0] + door_y2[i]*ortho_line[1] #()

                    overlap = (projected_room_1-projected_door_1) * (projected_room_2-projected_door_2) < 0
                    if any(overlap):
                        # assert only one overlapping
                        if not np.sum(overlap) == 1:
                            keep_door_line[i] = False
                            continue

                        # there is overlap
                        # keep the non overlapping part of the room line,
                        # cut the overlapped room line into two pieces

                        # NOTE: next boolean indexing assignment does not work
                        # backpropagte the indexing
                        theta_pass[theta_pass] = overlap
                        rho_pass[rho_pass] = theta_pass
                        
                        if np.abs(projected_room_1[overlap] - projected_door_1) < np.abs(projected_room_1[overlap] - projected_door_1):
                            # room_1 -> door_1, door_2-> room_2
                            # modify to room_1->door_1            projected_room_2 = None #(Nrp, )
                            room_x2[rho_pass] = door_x1[i]
                            room_y2[rho_pass] = door_y1[i]

                            # append the second
                            room_x1 = np.append(room_x1, door_x2[i])
                            room_y1 = np.append(room_y1, door_y2[i])

                            room_x2 = np.append(room_x2, passed_room_x2[overlap])
                            room_y2 = np.append(room_y2, passed_room_y2[overlap])

                            # append the hough representation
                            room_rhos = np.append(room_rhos, door_rho)
                            room_thetas = np.append(room_thetas, door_theta)


                        else:
                            # room_1 -> door_2, door_1-> room_2
                            # modify to room_1->door_1
                            room_x2[rho_pass] = door_x2[i]
                            room_y2[rho_pass] = door_y2[i]

                            # append the second
                            room_x1 = np.append(room_x1, door_x1[i])
                            room_y1 = np.append(room_y1, door_y1[i])

                            room_x2 = np.append(room_x2, passed_room_x2[overlap])
                            room_y2 = np.append(room_y2, passed_room_y2[overlap])

                            # append the hough representation
                            room_rhos = np.append(room_rhos, door_rho)
                            room_thetas =np.append(room_thetas, door_theta)

                        # disregard the door line 
                        keep_door_line[i] = False

                        if visualize:
                            plt.figure(1)
                            plt.axis("equal")
                            plt.plot([room_x1[:-1][rho_pass], room_x2[:-1][rho_pass]], [room_y1[:-1][rho_pass],room_y2[:-1][rho_pass]], 'b')
                            plt.pause(0.01)
                            plt.plot([room_x1[-1], room_x2[-1]], [room_y1[-1],room_y2[-1]], 'kx')
                            plt.pause(0.01)
                            plt.plot([door_x1[i], door_x2[i]], [door_y1[i],door_y2[i]], 'r')
                            plt.pause(0.01)


            else:
                # door line not overlapping with room line
                # keep the door line
                continue


    # plot remaining room_lines and kept door_line
    resolution = 0.02
    scale = 1/resolution
    x_min = x_min * scale
    y_min = y_min * scale
    x_max = x_max * scale
    y_max = y_max * scale
    H = int(y_max-y_min)
    W = int(x_max-x_min)

    if do_map:
        # fig = plt.figure()
        room_map = np.ones([H, W])*255
        # plt.axis("equal")
        for i in range(len(room_x1)):
            # plt.plot(line[:,0], line[:,1], "b")
            cv2.line(room_map, tuple((int(room_x1[i]*scale-x_min), int(room_y1[i]*scale-y_min))), tuple((int(room_x2[i]*scale-x_min), int(room_y2[i]*scale-y_min))), (0,255,0),1)
        # plt.imshow(room_map, cmap="gray", vmin=0, vmax=255)
        # plt.pause(0.01)

        for i in range(len(keep_door_line)):
            if keep_door_line[i]:
                cv2.line(room_map, tuple((int(door_x1[i]*scale-x_min), int(door_y1[i]*scale-y_min))), tuple((int(door_x2[i]*scale-x_min), int(door_y2[i]*scale-y_min))), (0,255,0),1)
        
        if save:
            cv2.imwrite(os.path.join(data_dir, scene, "map.png"), room_map)
        if visualize:
            plt.figure("room map")
            plt.clf()
            plt.imshow(room_map, origin="lower", cmap="gray", vmin=0, vmax=255)
            plt.pause(0.01)
        # print(time.time()-start_time)



    # process the pose from se3 to map se2
    # get poses
    poses_file = os.path.join(data_dir, scene, "poses.txt")

    # read poses
    with open(poses_file, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]

    poses_map = []
    # iterate through poses
    for i, pose_txt in enumerate(poses_txt): 
        pose = pose_txt.split(" ")
        x = float(pose[0])/1000 #pos was in mm 
        y = float(pose[1])/1000
        x = (x*scale-x_min)
        y = (y*scale-y_min)
        
        # x axis forward facing
        tx = float(pose[3])
        ty = float(pose[4])
        tz = float(pose[5])

        t = np.array([tx, ty, tz]) 
        t = t/np.linalg.norm(t)

        # z axis upwards
        ux = float(pose[6])
        uy = float(pose[7])
        uz = float(pose[8])

        u = np.array([ux, uy, uz])
        u = u/np.linalg.norm(t)
        
        # get the y axis
        w = np.cross(u, t)

        u = np.cross(t, w)

        # build the rotation matrix
        R = np.stack([t, w, u], axis=1)
        
        # Convert rotation matrix to a Rotation object
        r = Rotation.from_matrix(R)


        # Get Euler angles in radians
        th, pitch, roll = list(r.as_euler('ZYX'))  # 'zyx' represents the order of rotations

        pitch = pitch

        # get the image, check if correct
        image = cv2.imread(os.path.join(data_dir, scene, "imgs", str(i).zfill(3)+".png"))
        # hardcoded intrinsics from fov
        K = np.array([[320/np.tan(0.698132), 0, 320],
                    [0, 180/np.tan(0.440992), 180],
                    [0, 0, 1]], dtype=np.float32)
    
        image = gravity_align(image, r=roll, p=pitch, K=K, visualize=False)


        # write the poses
        pose_map = np.array([x, y, th, roll, pitch])
        poses_map.append(pose_map)
        if visualize:
            plt.figure("room_map")
            plt.clf()
            plt.imshow(room_map, origin="lower")
            plt.quiver([x], [y], [np.cos(th)], [np.sin(th)], angles="xy")
            plt.pause(0.1)

            plt.figure("whole")
            plt.quiver([float(pose[0])/1000], [float(pose[1])/1000], [np.cos(th)], [np.sin(th)], angles="xy")
            plt.pause(0.1)

    if save:
        # poses in world frame and in map frame
        with open(os.path.join(data_dir, scene, "poses_map.txt"), "w") as f:
            for pose_map in poses_map:
                # writeline to pf traj file
                f.write("{} {} {} {} {}\n".format(pose_map[0], pose_map[1], pose_map[2], pose_map[3], pose_map[4]))     


