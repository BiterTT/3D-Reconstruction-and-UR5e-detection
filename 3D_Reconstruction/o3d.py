import open3d as o3d
import numpy as np

# 加载点云
pcd = o3d.io.read_point_cloud("res/milk3.ply")

# 转换为 numpy 数组
points = np.asarray(pcd.points)



z_threshold = 0.0
mask = points[:, 2] > z_threshold

points_half = points[mask]


points_mirror = points_half.copy()
points_mirror[:, 0] *= -1  # X轴翻转
points_mirror[:, 2] *= -1  # Z轴翻转


merged_points = np.vstack((points_half, points_mirror))

pcd_combined = o3d.geometry.PointCloud()
pcd_combined.points = o3d.utility.Vector3dVector(merged_points)
# 设置阈值
# x_threshold = 100000.0
# y_threshold = 100000.0
# z_threshold = 100000.0

# x_threshold = -10.0
# y_threshold = -10.0
# z_threshold = 0.0
# z_max_threshold = 400
# # 组合过滤条件：三个方向都要大于阈值
# # filtered_indices = (points[:, 0] > x_threshold) & \
# #                    (points[:, 1] > y_threshold) & \
# #                    (points[:, 2] > z_threshold)
# filtered_indices =  ((points[:, 2]<z_max_threshold))
# # 应用过滤
# filtered_points = points[filtered_indices]
# pcd.points = o3d.utility.Vector3dVector(filtered_points)

# # 如果有颜色，也同步过滤
# if pcd.has_colors():
#     colors = np.asarray(pcd.colors)
#     pcd.colors = o3d.utility.Vector3dVector(colors[filtered_indices])

# 可视化或保存
o3d.visualization.draw_geometries([pcd_combined])
o3d.io.write_point_cloud("filtered_point_cloud.ply", pcd)
