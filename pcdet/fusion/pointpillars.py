import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import io
EPSILON = 1e-6


def make_grid(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset
    xgrid_res, ygrid_res, zgrid_res = grid_res

    xcoords = torch.arange(0, depth+xgrid_res, xgrid_res) + xoff
    ycoords = torch.arange(0, width+ygrid_res, ygrid_res) + yoff

    xx, yy = torch.meshgrid(xcoords, ycoords)
    grid = torch.stack([xx, yy, torch.full_like(xx, zoff)], dim=-1)
    return grid.unsqueeze(0)


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


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)


def oft(features, calib, grid, r_rect, velo2cam, grid_height, cell_size):  # grid x y z

    y_corners = torch.arange(0, grid_height+cell_size, cell_size)  # [0,-2,0]...[0,2,0]
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [2, 0])
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)  # 1x9x433(x)x497(y)x3

    batch_size, z, x, y, _ = corners.size()
    corners = corners.view(-1, 3)
    corners = corners.numpy()

    corners = lidar_to_camera(corners, r_rect, velo2cam)

    img_corners = project_to_image(corners, calib)
    img_corners = torch.from_numpy(img_corners)
    img_corners = img_corners.view(batch_size, z,  x, y,   -1)

    corners = torch.from_numpy(corners)
    corners = corners.view(batch_size, z, x, y, -1)
    # Project grid corners to image plane and normalize to [-1, 1]
    # img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)  # 1x8x160x160x2
    # features = (features - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
    # Normalize to [-1, 1]
    img_height, img_width = features.size()[2:]
    img_size = corners.new([img_width, img_height])
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)  # 1x8x160x160x2

    # Get top-left and bottom-right coordinates of voxel bounding boxes
    bbox_corners = torch.cat([
        torch.min(norm_corners[:, :-1, :-1, :-1],  # 1x7x159x159x2
                    norm_corners[:, :-1, 1:,:-1]),
        torch.max(norm_corners[:, 1:, 1:, 1:],
                    norm_corners[:, 1:,:-1,1:])
    ], dim=-1,).float()  # 1x7x159x159x4
    batch, _, depth, width, _ = bbox_corners.size()
    bbox_corners = bbox_corners.flatten(2, 3)  # 1x7x25281x4

    # Compute the area of each bounding box
    # area:1x1x159x1113
    area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1)
            * img_height * img_width * 0.25 + EPSILON).unsqueeze(1)  # 1x1x7x25281
    visible = (area > EPSILON)

    # Sample integral image at bounding box locations
    integral_img = integral_image(features).float()  # 1x3x370x1224
    top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])  # 1x3x7x25281
    btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])
    top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])
    btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])

    # Compute voxel features (ignore features which are not visible)
    vox_feats = (top_left + btm_right - top_right - btm_left) / area  # 1x3x7x25281
    vox_feats = vox_feats * visible.float()

    # vox_feats = vox_feats.view(batch, -1, depth, width)
    vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)  # 73408 x24
    vox_feats = vox_feats.view(1, y-1, x-1 , -1)  # H x W x 24
    vox_feats = vox_feats.permute(0, 3, 1, 2)
    vox_feats = vox_feats.squeeze(0)
    vox_feats = vox_feats.numpy()
    vox_feats = vox_feats.astype(np.float32)
    return vox_feats


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(path, 'testing/image_2')
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
        calib_file = os.path.join(path, 'testing/calib')
        calib_file = os.path.join(calib_file, calib_id)
        seg_score_id = img_num + '.bin'
        seg_score_file = os.path.join(path, 'testing/seg_score')
        seg_score_file = os.path.join(seg_score_file, seg_score_id)
        seg_score = np.fromfile(str(seg_score_file), dtype=np.float32, count=-1).reshape([-1, h, w])
        seg_score = np.argmax(seg_score, axis=0)
        seg_score = seg_score[np.newaxis, :, :] / 3
        seg_score = seg_score.astype(np.float32)

        image = torch.from_numpy(seg_score)
        image = image.unsqueeze(0)

        calib = read_kitti_calib(calib_file)
        calib = calib.astype(np.float32)

        r_rect = read_kitti_R0_rect(calib_file)
        r_rect = r_rect.astype(np.float32)

        velo2cam = read_kitti_Tr_velo_to_cam(calib_file)
        velo2cam = velo2cam.astype(np.float32)
        grid = make_grid((47.36, 39.68), (0, -19.84, -2.5), (0.16, 0.16, 0.5))  # 160x160x3 [-40,1.74,0] [-39.5,1.74,0]
        trans = oft(image, calib, grid, r_rect, velo2cam, 3, 0.5)
        save_id = img_num + '.npy'
        save_path = os.path.join(path, 'testing/oft_class')
        save_path = os.path.join(save_path, save_id)
        np.save(save_path,trans)
        print(img_num)


if __name__ == '__main__':
    # main()
    a = np.load('/home/yyl/Desktop/OFT/training/oft_class/000000.npy')
    print(a)
    print(a.shape)
