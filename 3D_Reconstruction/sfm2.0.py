import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt

class Image_loader():
    def __init__(self, img_dir:str, downscale_factor:float):
        # loading the Camera intrinsic parameters K
        # self.K = np.array([[1342.804667, 0, 903.751193],
        #             [0, 1317.430995, 514.675065],
        #             [0, 0, 1]])
        self.K = np.array([[696.368502, 0, 277.548495],
                    [0, 704.218753, 275.653200],
                    [0, 0, 1]])
        #1080p yaopeng
        # self.K = np.array([[812.170264, 0, 367.584947],
        #             [0, 822.744764, 191.863713],
        #             [0, 0, 1]])
        # self.K = np.array([[747.59979, 0, 298.890599],
        #             [0, 746.77745, 227.979219],
        #             [0, 0, 1]])
        # with open(img_dir + '/K.txt') as f:
        #     self.K = np.array(list((map(lambda x:list(map(lambda x:float(x), x.strip().split(' '))),f.read().split('\n')))))
        self.image_list = []
        # Loading the set of images
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(img_dir + '/' + image)
        
        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale()
    
    def downscale(self) -> None:
        '''
        Downscales the Image intrinsic parameter acc to the downscale factor
        '''
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor
    
    def downscale_image(self, image):
        for _ in range(1,int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image

class Sfm():
    def __init__(self, img_dir:str, downscale_factor:float = 1.0) -> None:
        '''
            Initialise and Sfm object.
        '''
        self.img_obj = Image_loader(img_dir,downscale_factor)

    #triangulation
    #P = K [R | t] → 3x4, projection_matrix_1, projection_matrix_1
    def triangulation(self, projection_matrix_1, projection_matrix_2, point_2d_1, point_2d_2) -> tuple:
        '''
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, point cloud 
        '''
        print("projection_matrix_1:")
        print(projection_matrix_1.shape)
        print(point_2d_1.shape)

        pt_cloud_points = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, point_2d_1.T, point_2d_2.T)
        pt_cloud_points = (pt_cloud_points / pt_cloud_points[3])

        return point_2d_1.T, point_2d_2.T, pt_cloud_points      
    
    def PnP(self, obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:
        '''
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        '''
        if initial == 1:
            obj_point = obj_point[:, 0 ,:]
            image_point = image_point.T
            rot_vector = rot_vector.T 
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)
        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector
    
    def convertPoints_Homogeneous(self, obj_points,homogenity) ->tuple:
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        return obj_points

    def to_ply(self, path, point_cloud, colors) -> None:
        out_points = point_cloud.reshape(-1, 3) * 0.0000000000000001
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        # 输出点范围信息（新增）
        print("\n--- Point Cloud Statistics ---")
        print(f"X range: {np.min(verts[:, 0]):.2f} ~ {np.max(verts[:, 0]):.2f}")
        print(f"Y range: {np.min(verts[:, 1]):.2f} ~ {np.max(verts[:, 1]):.2f}")
        print(f"Z range: {np.min(verts[:, 2]):.2f} ~ {np.max(verts[:, 2]):.2f}")
        print(f"Mean (center): X={np.mean(verts[:, 0]):.2f}, Y={np.mean(verts[:, 1]):.2f}, Z={np.mean(verts[:, 2]):.2f}")

        # 居中与筛选
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)

        # 输出距离信息（新增）
        print(f"Distance to center - mean: {np.mean(dist):.2f}, max: {np.max(dist):.2f}, std: {np.std(dist):.2f}")

        # 筛选
        threshold = np.mean(dist) +1000
        indx = np.where(dist < threshold)
        verts = verts[indx]

        print(f"Filtered points: {len(verts)} / {len(out_points)} retained (threshold = {threshold:.2f})")

        # 写入ply
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
        with open(path + '/res/' + self.img_obj.image_list[0].split('/')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')


    # image_points_1: image_previous_match_keypoints 3D---2D
    # image_points_2: image_current_keypoints
    # image_points_3: image_next_keypoints
    def find_same_points(self, image_previous_match_keypoints, image_current_keypoints, image_next_keypoints) -> tuple:
        '''
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3 
        '''
        common_current_point = []
        common_next_point = []
        for i in range(image_previous_match_keypoints.shape[0]):
            same_flag = np.where(image_current_keypoints == image_previous_match_keypoints[i, :])
            if same_flag[0].size != 0:
                common_current_point.append(i)
                common_next_point.append(same_flag[0][0])
        
        mask_array_1 = np.ma.array(image_current_keypoints, mask=False)
        mask_array_1.mask[common_next_point] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_next_keypoints, mask=False)
        mask_array_2.mask[common_next_point] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(common_current_point), np.array(common_next_point), mask_array_1, mask_array_2

    #detect and matches
    def find_features(self, image_0, image_1):
        # mask = np.zeros((1080,1920), dtype=np.uint8)
        # sift = cv2.SIFT_create(contrastThreshold=0.001, edgeThreshold=20)
        sift = cv2.SIFT_create(contrastThreshold=0.0001, edgeThreshold=20)#
        #mask[0:1000, 650:1500] = 255
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.95 * n.distance:
                feature.append(m)
        image_0_keypoints = np.float32([key_points_0[m.queryIdx].pt for m in feature])   #(N,2)
        image_1_keypoints = np.float32([key_points_1[m.trainIdx].pt for m in feature])   #(N,2)
        return image_0_keypoints, image_1_keypoints
    #estimate_motion
    def estimate_motion(self, kp1 , kp2 , K):
        E, mask = cv2. findEssentialMat (kp1 , kp2 , K, method = cv2. RANSAC , prob =0.999 ,threshold =0.5)
        kp1 = kp1[mask.ravel() == 1]
        kp2 = kp2[mask.ravel() == 1]
        _, R, t, mask = cv2. recoverPose (E, kp1 , kp2 , K)
        kp1 = kp1[mask.ravel() > 0]
        kp2 = kp2[mask.ravel() > 0]   
        return R, t, kp1, kp2
    
    def __call__(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel()
        #world matrix
        transform_matrix_0 = np.array([[1, 0, 0, 0], 
                               [0, 1, 0, 0], 
                               [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
    
        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4)) 
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        image_0_keypoints, image_1_keypoints = self.find_features(image_0, image_1)
        # Essential matrix
        # Calculate the relative pose of the first two frames.
        R, t, image_0_keypoints, image_1_keypoints = self.estimate_motion(image_0_keypoints, image_1_keypoints, self.img_obj.K)
        transform_matrix_1[:3, :3] = np.matmul(R, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], t.ravel())
        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)
        #Initial point cloud of triangulation
        image_0_keypoints, image_1_keypoints, points_3d = self.triangulation(pose_0, pose_1, image_0_keypoints, image_1_keypoints)
        points_3d = self.convertPoints_Homogeneous(points_3d, homogenity = 1)
        #Re-estimate pose_1 using PnP
        _, _, image_1_keypoints, points_3d, _ = self.PnP(points_3d, image_1_keypoints, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), image_0_keypoints, initial=1)

        total_images = len(self.img_obj.image_list) - 2 
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        image_current = image_1
        image_previous = image_0
        image_previous_match_keypoints = image_1_keypoints
        for i in (range(total_images)):
            image_next = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))#img2
            image_current_keypoints, image_next_keypoints = self.find_features(image_current, image_next) #img1 img2

            if i != 0:
                #caculate image 0 and image 1
                image_0_keypoints, image_1_keypoints, points_3d = self.triangulation(pose_0, pose_1, image_0_keypoints, image_1_keypoints)
                image_1_keypoints = image_1_keypoints.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.find_same_points(image_1_keypoints, image_current_keypoints, image_next_keypoints)
            cm_points_2 = image_next_keypoints[cm_points_1]
            cm_points_cur = image_current_keypoints[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            points_3d = self.convertPoints_Homogeneous(points_3d, homogenity = 0)
        
            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            points_3d = self.convertPoints_Homogeneous(points_3d, homogenity = 1)
            pose_array = np.hstack((pose_array, pose_2.ravel()))

            total_points = np.vstack((total_points, points_3d[:, 0, :]))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([image_next[l[1], l[0]] for l in points_left.T])
            total_colors = np.vstack((total_colors, color_vector)) 
   
            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            #plt.scatter(i, error)
            plt.pause(0.05)

            image_previous = np.copy(image_current)
            image_current = np.copy(image_next)
            image_0_keypoints = np.copy(image_current_keypoints)
            image_1_keypoints = np.copy(image_next_keypoints)
            pose_1 = np.copy(pose_2)
            cv2.imshow(self.img_obj.image_list[0].split('/')[-2], image_next)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '/res/' + self.img_obj.image_list[0].split('/')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')

if __name__ == '__main__':
    # sfm = Sfm("Datasets/Herz-Jesus-P8")
    sfm = Sfm("Datasets/milk3")
    sfm()
