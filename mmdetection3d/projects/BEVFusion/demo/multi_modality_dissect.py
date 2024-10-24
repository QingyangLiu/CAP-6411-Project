# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

from mmdet3d.apis import inference_multi_modality_detector, init_model, inference_multi_modality_detector_feature
from mmdet3d.registry import VISUALIZERS

import torch
from torchvision.ops import roi_pool
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('img', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM_FRONT',
        help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def project_3d_to_2d_simple(coordinates_3d, feature_map_size):
    """
    Project 3D coordinates to 2D feature map space using simple perspective projection.
    
    Args:
        coordinates_3d: Tensor of shape (N, 7) containing [x, y, z, dx, dy, dz, yaw]
        feature_map_size: Tuple of (H, W) of the feature map
    
    Returns:
        boxes_2d: Tensor of shape (N, 4) containing [x1, y1, x2, y2] in feature map coordinates
    """
    # Extract center coordinates and dimensions
    centers = coordinates_3d[:, :3]  # (N, 3) - [x, y, z]
    dimensions = coordinates_3d[:, 3:6]  # (N, 3) - [dx, dy, dz]
    yaws = coordinates_3d[:, 6]  # (N,) - rotation around z-axis
    
    # Create 8 corners of the 3D box
    l, w, h = dimensions[:, 0:1], dimensions[:, 1:2], dimensions[:, 2:3]
    
    # Define the 8 corners of a 3D bounding box
    x_corners = torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)
    y_corners = torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)
    z_corners = torch.cat([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)
    
    # Rotate according to yaw
    cosa = torch.cos(yaws)
    sina = torch.sin(yaws)
    
    # Rotation matrix around z-axis
    R = torch.stack([
        torch.stack([cosa, -sina, torch.zeros_like(cosa)], dim=1),
        torch.stack([sina, cosa, torch.zeros_like(cosa)], dim=1),
        torch.stack([torch.zeros_like(cosa), torch.zeros_like(cosa), torch.ones_like(cosa)], dim=1)
    ], dim=1)
    
    # Rotate corners
    corners = torch.stack([x_corners, y_corners, z_corners], dim=2)
    corners = torch.bmm(corners, R.transpose(1, 2))
    
    # Translate to center position
    corners = corners + centers.unsqueeze(1)
    
    # Simple perspective projection (assuming camera at origin looking along z-axis)
    H, W = feature_map_size
    scale_factor = 1.0  # Adjust this based on your scene scale
    
    projected_x = corners[:, :, 0] / (corners[:, :, 2] * scale_factor + 1e-6)
    projected_y = corners[:, :, 1] / (corners[:, :, 2] * scale_factor + 1e-6)
    
    # Scale to feature map size
    projected_x = (projected_x + 1) * W / 2
    projected_y = (projected_y + 1) * H / 2
    
    # Get 2D bounding box from projected corners
    x1 = projected_x.min(dim=1)[0]
    y1 = projected_y.min(dim=1)[0]
    x2 = projected_x.max(dim=1)[0]
    y2 = projected_y.max(dim=1)[0]
    
    boxes_2d = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes_2d

def crop_and_pool_features(feature_map, coordinates_3d, output_size=(7, 7)):
    """
    Crops and pools features from a feature map using 3D object coordinates.
    
    Args:
        feature_map: Tensor of shape (1, 512, 180, 180)
        coordinates_3d: Tensor of shape (200, 7) containing [x, y, z, dx, dy, dz, yaw]
        output_size: Tuple of (height, width) for the output pooled features
    
    Returns:
        pooled_features: Tensor of shape (200, 512) containing pooled features for each object
    """
    feature_map_size = (feature_map.shape[2], feature_map.shape[3])
    
    # Project 3D boxes to 2D feature map space
    boxes_2d = project_3d_to_2d_simple(coordinates_3d, feature_map_size)
    
    # Clip coordinates to valid range
    boxes_2d[:, [0, 2]] = torch.clamp(boxes_2d[:, [0, 2]], 0, feature_map_size[1])
    boxes_2d[:, [1, 3]] = torch.clamp(boxes_2d[:, [1, 3]], 0, feature_map_size[0])
    
    # Normalize coordinates to [0, 1] range
    boxes_2d = boxes_2d.float()
    boxes_2d[:, [0, 2]] /= feature_map_size[1]  # normalize x coordinates
    boxes_2d[:, [1, 3]] /= feature_map_size[0]  # normalize y coordinates
    
    # Add batch index for each box
    batch_idx = torch.zeros(len(boxes_2d), 1, device=boxes_2d.device)
    rois = torch.cat([batch_idx, boxes_2d], dim=1)
    
    # Use RoI pooling from torchvision.ops
    pooled_features = roi_pool(
        feature_map, 
        rois,
        output_size=output_size,
        spatial_scale=1.0
    )
    
    # Global average pooling to get final feature vector
    pooled_features = pooled_features.mean(dim=[2, 3])  # (200, 512)
    
    return pooled_features








def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image and point cloud sample
    result, data, feats = inference_multi_modality_detector_feature(model, args.pcd, args.img,
                                                     args.ann, args.cam_type)
    points = data['inputs']['points']

    print("#####")
    print(type(data))
    print("#####")
    print(type(result))
    print("#####")
    print(type(feats))
    print("#####")
    #print(feats)
    print("#####")




    # For predicted bounding boxes
    if 'pred_instances_3d' in result:
        pred_instances = result.pred_instances_3d
        print(pred_instances)
        print("#####")
        if 'bboxes_3d' in pred_instances:
            print("Predicted 3D bounding boxes:")
            print(pred_instances.bboxes_3d)
            print("#####") 
            print(pred_instances.bboxes_3d.tensor) # (x, y, z, dx, dy, dz, yaw, vx, vy)
            print((pred_instances.bboxes_3d.tensor).shape)
            print((pred_instances.scores_3d).shape)

            # #Find Minx and Max Value
            # min_value = torch.min(pred_instances.bboxes_3d.tensor)
            # max_value = torch.max(pred_instances.bboxes_3d.tensor)
            # print(min_value)
            # print(max_value)

            #
            # Step 1: Remove vx and vy (last two columns)
            boxes_3d_reduced = pred_instances.bboxes_3d.tensor[:, :7]  # Now boxes_3d_reduced is (200, 7)
            print(boxes_3d_reduced.shape)
            print(boxes_3d_reduced.shape[0])

            ##### Step 2, Get specific (1, 512) for object through some operations(Now, easy part)
            ##### (later follow exact paper)

            #feature_map = torch.randn(1, 512, 180, 180)
            #coordinates_3d = torch.randn(200, 7)  # [x, y, z, dx, dy, dz, yaw]
            # Get pooled features
            #pooled_features = crop_and_pool_features(feature_map, coordinates_3d)

            pooled_features = crop_and_pool_features(feats, boxes_3d_reduced)
            
            print(f"Pooled features shape: {pooled_features.shape}")  # Should be (200, 512)
            

            ###Select 130 objects based on some criterion, such as confidence score
            ###Suppose you have a tensor 'scores' representing the confidence scores for each box
            scores = pred_instances.scores_3d
            if (boxes_3d_reduced.shape[0] > 130):
                #top_indices = scores.topk(130).indices
                top_indices = pred_instances.scores_3d.topk(130).indices
                boxes_3d_final = boxes_3d_reduced[top_indices]
                pooled_features_final = pooled_features[top_indices]
            else:
                boxes_3d_final = boxes_3d_reduced
                pooled_features_final = pooled_features
                

            #boxes_3d_final (Most confident bounding boxes)

            print(f"boxes_3d_final shape: {boxes_3d_final.shape}")
            print(f"pooled_features_final shape: {pooled_features_final.shape}")


            

    end

    # # For predicted bounding boxes
    # if 'pred_instances_3d' in results:
    #     pred_instances = results.pred_instances_3d
    #     if 'bboxes_3d' in pred_instances:
    #         print("Predicted 3D bounding boxes:")
    #         print(pred_instances.bboxes_3d)

    # # For ground truth bounding boxes
    # if 'gt_instances_3d' in results:
    #     gt_instances = results.gt_instances_3d
    #     if 'bboxes_3d' in gt_instances:
    #         print("Ground truth 3D bounding boxes:")
    #         print(gt_instances.bboxes_3d)


    print("#####")
    print(type(points))
    print(points.shape)
    print("#####")
    
    #end

    if isinstance(result.img_path, list):
        img = []
        for img_path in result.img_path:
            single_img = mmcv.imread(img_path)
            single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
            img.append(single_img)
    else:
        img = mmcv.imread(result.img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
    data_input = dict(points=points, img=img)

    # show the results
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=-1,
        out_file=args.out_dir,
        pred_score_thr=args.score_thr,
        vis_task='multi-modality_det')


if __name__ == '__main__':
    args = parse_args()
    main(args)
