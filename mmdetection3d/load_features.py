
#Model stuff
from nuscenes.nuscenes import NuScenes
from mmdet3d.apis import init_model, inference_detector
import os

#######################

# Load the NuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/imr555/Downloads/projects/mmdet3d_installation/mmdetection3d/data/nuscenes', verbose=True)

# Specify the sample token for the scene you want to run inference on
sample_token = 'e93e98b63d3b40209056d129dc53ceee'  # Replace with actual sample token
sample = nusc.get('sample', sample_token)

print("#####")
print(sample)
print("#####")

print("Sample Data#####")
#######
##{'token': 'e93e98b63d3b40209056d129dc53ceee', 'timestamp': 1531883530449377, 'prev': '', 'next': '14d5adfe50bb4445bc3aa5fe607691a8', 'scene_token': '73030fb67d3c46cfb5e590168088ae39', 'data': {'RADAR_FRONT': 'bddd80ae33ec4e32b27fdb3c1160a30e', 'RADAR_FRONT_LEFT': '1a08aec0958e42ebb37d26612a2cfc57', 'RADAR_FRONT_RIGHT': '282fa8d7a3f34b68b56fb1e22e697668', 'RADAR_BACK_LEFT': '05fc4678025246f3adf8e9b8a0a0b13b', 'RADAR_BACK_RIGHT': '31b8099fb1c44c6381c3c71b335750bb', 'LIDAR_TOP': '3388933b59444c5db71fade0bbfef470', 'CAM_FRONT': '020d7b4f858147558106c504f7f31bef', 'CAM_FRONT_RIGHT': '16d39ff22a8545b0a4ee3236a0fe1c20', 'CAM_BACK_RIGHT': 'ec7096278e484c9ebe6894a2ad5682e9', 'CAM_BACK': 'aab35aeccbda42de82b2ff5c278a0d48', 'CAM_BACK_LEFT': '86e6806d626b4711a6d0f5015b090116', 'CAM_FRONT_LEFT': '24332e9c554a406f880430f17771b608'}, 'anns': ['173a50411564442ab195e132472fde71', '5123ed5e450948ac8dc381772f2ae29a', 'acce0b7220754600b700257a1de1573d', '8d7cb5e96cae48c39ef4f9f75182013a', 'f64bfd3d4ddf46d7a366624605cb7e91', 'f9dba7f32ed34ee8adc92096af767868', '086e3f37a44e459987cde7a3ca273b5b', '3964235c58a745df8589b6a626c29985', '31a96b9503204a8688da75abcd4b56b2', 'b0284e14d17a444a8d0071bd1f03a0a2']}
##Break
##{'token': 'e93e98b63d3b40209056d129dc53ceee', 'timestamp': 1531883530449377, 'prev': '', 'next': '14d5adfe50bb4445bc3aa5fe607691a8', 'scene_token': '73030fb67d3c46cfb5e590168088ae39', 
#'data': {'RADAR_FRONT': 'bddd80ae33ec4e32b27fdb3c1160a30e', 'RADAR_FRONT_LEFT': '1a08aec0958e42ebb37d26612a2cfc57', 'RADAR_FRONT_RIGHT': '282fa8d7a3f34b68b56fb1e22e697668', 'RADAR_BACK_LEFT': '05fc4678025246f3adf8e9b8a0a0b13b', 'RADAR_BACK_RIGHT': '31b8099fb1c44c6381c3c71b335750bb', 'LIDAR_TOP': '3388933b59444c5db71fade0bbfef470', 'CAM_FRONT': '020d7b4f858147558106c504f7f31bef', 'CAM_FRONT_RIGHT': '16d39ff22a8545b0a4ee3236a0fe1c20', 'CAM_BACK_RIGHT': 'ec7096278e484c9ebe6894a2ad5682e9', 'CAM_BACK': 'aab35aeccbda42de82b2ff5c278a0d48', 'CAM_BACK_LEFT': '86e6806d626b4711a6d0f5015b090116', 'CAM_FRONT_LEFT': '24332e9c554a406f880430f17771b608'}, 'anns': ['173a50411564442ab195e132472fde71', '5123ed5e450948ac8dc381772f2ae29a', 'acce0b7220754600b700257a1de1573d', '8d7cb5e96cae48c39ef4f9f75182013a', 'f64bfd3d4ddf46d7a366624605cb7e91', 'f9dba7f32ed34ee8adc92096af767868', '086e3f37a44e459987cde7a3ca273b5b', '3964235c58a745df8589b6a626c29985', '31a96b9503204a8688da75abcd4b56b2', 'b0284e14d17a444a8d0071bd1f03a0a2']}
## 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP' (Maybe just takes top)
##
## 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
#######
print("######")


#######################


#######################

#Camera View Dictionary
camera_view_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
#Lidar View Dictionary
lidar_views_keys = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']


#######################


#######################

# #For one case

# #####
# # Get the camera image path (e.g., from the front camera)
# cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
# cam_path = os.path.join(nusc.dataroot, cam_data['filename'])


# #####
# # Get the lidar point cloud path
# lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
# lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

# #######################



# #######################

# #For multiple cases

# #####

# for camera_view in camera_view_keys:
#     # Get the camera image path (e.g., from the front camera)
#     cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
#     cam_path = os.path.join(nusc.dataroot, cam_data['filename'])

# for lidar_views in lidar_views_keys:
#     #####
#     # Get the lidar point cloud path
#     lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
#     lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

# #######################




#######################

#For required cases

#####

for camera_view in camera_view_keys:
    # Get the camera image path (e.g., from the front camera)
    cam_data = nusc.get('sample_data', sample['data'][camera_view])
    cam_path = os.path.join(nusc.dataroot, cam_data['filename'])
    
    print("#####")
    print(cam_path)
    print("#####")


print("##################################")

#####
# Get the lidar point cloud path
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

print("#####")
print(lidar_path)
print("#####")


#######################

