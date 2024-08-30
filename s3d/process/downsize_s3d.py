import os
import tqdm
import cv2

data_dir = "~/Downloads/Structured3D"
# get all scene name (dir) in the path
scenes = sorted(os.listdir(data_dir))

for scene in tqdm.tqdm(scenes):
    img_dir = os.path.join(data_dir, scene, "imgs")
    
    for img in sorted(os.listdir(img_dir)):
        n_img = img[:-4]
        im = cv2.imread(os.path.join(img_dir, img))
        im = cv2.resize(im, (640, 360))
        cv2.imwrite(os.path.join(img_dir, n_img.zfill(3)+".png"), im)
        os.system("rm "+os.path.join(img_dir, img))