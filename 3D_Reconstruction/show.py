import open3d as o3d
import numpy as np
import numpy as np

def rotate_points(points, axis='z', angle_deg=90):
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    
    return points @ R.T  # 注意转置

# 加载原始点云
pcd = o3d.io.read_point_cloud("res/milk3.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Step1️⃣：选取正确的一半（三角形1）

#mask filter
x_threshold = 500.0
y_threshold = 500.0
z_threshold = 0.0
z_max_threshold = 350
x_min_threshold = 0
y_min_threshold = -400
mask = (points[:, 0] < x_threshold) & \
        (points[:, 1] < y_threshold) & \
        (points[:, 2] > z_threshold) & \
        (points[:, 2] < z_max_threshold) &\
        (points[:, 0] > x_min_threshold) &\
        (points[:, 1] > y_min_threshold)
z_threshold = 0.0  # 你可以根据图像调整
# mask = points[:, 2] > z_threshold
# mask = points[:, 1] < 150
points_half = points[mask]
# colors_half = colors[mask]

# Step2️⃣：复制 + 翻转出镜像副本（三角形2）
points_mirror = points_half.copy()
points_mirror[:, 0] *= -1    # X轴镜像
points_mirror[:, 2] *= -1    # Z轴镜像（斜向）
points_mirror = rotate_points(points_mirror, axis='z', angle_deg=10)

# Step3️⃣：调整平移让它“拼上去” ——————
# ↓↓↓ 修改下面的平移量
offset = np.array([400, 0, 400.0]) 
points_mirror += offset

# Step4️⃣：拼接两个三角形点云
merged_points = np.vstack((points_half, points_mirror))
# merged_colors = np.vstack((colors_half, colors_half))

# Step5️⃣：生成新点云
pcd_combined = o3d.geometry.PointCloud()
pcd_combined.points = o3d.utility.Vector3dVector(merged_points)
# pcd_combined.colors = o3d.utility.Vector3dVector(merged_colors)


# 添加坐标轴辅助线（1米长）
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000)

# 同时显示点云和坐标轴
o3d.visualization.draw_geometries([pcd_combined, axis])

# 可视化
# o3d.visualization.draw_geometries([pcd_combined])
o3d.io.write_point_cloud("merged_triangles.ply", pcd_combined)
