import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage import io
import cv2

def create_anchors_3d_stride(grid_size,
                             voxel_size=[0.16, 0.16, 0.5],
                             coordinates_offsets=[0, -19.84, -2.5],

                             dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz
    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride, z_stride = voxel_size
    x_offset, y_offset, z_offset = coordinates_offsets
    x_centers = np.arange(grid_size[0], dtype=dtype)
    y_centers = np.arange(grid_size[1], dtype=dtype)
    z_centers = np.arange(grid_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset + 0.25

    y_centers = y_centers * y_stride + y_offset + 0.08
    x_centers = x_centers * x_stride + x_offset + 0.08

    xx, yy, zz  = np.meshgrid(x_centers, y_centers, z_centers)

    sizes = np.stack((xx, yy , zz), axis=-1)

    sizes = np.reshape(sizes, [-1,3])

    return sizes


def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])  # N
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def project_to_image(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def read_kitti_calib(filename):
    """Read the camera 2 calibration matrix from a text file"""

    with open(filename) as f:
        for line in f:
            data = line.split(' ')
            if data[0] == 'P2:':
                calib_P2 = np.array([float(x) for x in data[1:13]])
                calib_P2 = calib_P2.reshape(3, 4)
                return _extend_matrix(calib_P2)

    raise Exception(
        'Could not find entry for P2 in calib file {}'.format(filename))


def read_kitti_R0_rect(filename):
    """Read the camera 2 calibration matrix from a text file"""

    with open(filename) as f:
        for line in f:
            data = line.split(' ')
            if data[0] == 'R0_rect:':
                R0_rect = np.array([float(x) for x in data[1:13]])
                R0_rect = R0_rect.reshape(3,3)
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
                return rect_4x4

    raise Exception(
        'Could not find entry for P2 in calib file {}'.format(filename))


def read_kitti_Tr_velo_to_cam(filename):
    """Read the camera 2 calibration matrix from a text file"""

    with open(filename) as f:
        for line in f:
            data = line.split(' ')
            if data[0] == 'Tr_velo_to_cam:':
                calib = np.array([float(x) for x in data[1:13]])
                calib = calib.reshape(3, 4)
                return _extend_matrix(calib)

    raise Exception(
        'Could not find entry for P2 in calib file {}'.format(filename))


def oft(features, calib,  r_rect, velo2cam):  # grid x y z
    H, W = features.shape
    features = np.swapaxes(features, 0,1)

    corners = create_anchors_3d_stride([296,248,6])

    corners = lidar_to_camera(corners, r_rect, velo2cam)

    points_v_to_image = project_to_image(corners, calib)
    num_points = len(points_v_to_image)
    pointing = np.zeros(num_points)


    for num in range(num_points):
        u = int(round(points_v_to_image[num][0]))
        v = int(round(points_v_to_image[num][1]))
        if  -1 < u < W and -1< v < H:
            pointing[num] = features[u][v]

    pointing = pointing.reshape(248,296,6) # 248 296 8 3
    pointing = np.swapaxes(pointing,0,2)
    pointing = np.swapaxes(pointing, 1, 2)
    pointing= pointing.astype(np.float32)
    return pointing



def main():
    path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(path, 'training/image_2')
    data_dir = image_path
    images = os.listdir(data_dir)
    if len(images) == 0:
        print('There are no images at directory %s. Check the data path.' % (data_dir))
    else:
        print('There are %d images to be processed.' % (len(images)))
    images.sort()
    for img_id, img_name in enumerate(images):

        img_dir = os.path.join(data_dir, img_name)

        img_shape = np.array(
            io.imread(img_dir).shape[:2], dtype=np.int32)
        h, w = np.array(img_shape, dtype=np.int32)
        img_num = img_name.split(".png")[0]
        calib_id = img_num + '.txt'
        calib_file = os.path.join(path, 'training/calib')
        calib_file = os.path.join(calib_file, calib_id)
        seg_score_id = img_num + '.bin'
        seg_score_file = os.path.join(path, 'training/seg_score')
        seg_score_file = os.path.join(seg_score_file, seg_score_id)
        seg_score = np.fromfile(str(seg_score_file), dtype=np.float32, count=-1).reshape([-1, h, w])
        seg_score = np.argmax(seg_score, axis=0)
        #seg_score = seg_score[np.newaxis, :, :]
        seg_score = seg_score.astype(np.float32)



        calib = read_kitti_calib(calib_file)
        calib = calib.astype(np.float32)

        r_rect = read_kitti_R0_rect(calib_file)
        r_rect = r_rect.astype(np.float32)

        velo2cam = read_kitti_Tr_velo_to_cam(calib_file)
        velo2cam = velo2cam.astype(np.float32)


        trans = oft(seg_score, calib,  r_rect, velo2cam)
        save_id = img_num + '.npy'
        save_path = os.path.join(path, 'training/oft_class')
        save_path = os.path.join(save_path, save_id)
        np.save(save_path,trans)
        print(img_num)


if __name__ == '__main__':
    main()