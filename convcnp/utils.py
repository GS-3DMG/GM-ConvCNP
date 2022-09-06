import math
import os

import numpy as np


def cast_numpy(tensor):
    if not isinstance(tensor, np.ndarray):
        return tensor.data.cpu().numpy()
        # return tensor.numpy()
    return tensor


def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)


def channel_first(x):
    return x.transpose(3, 2).transpose(2, 1)


def channel_last_3d(x):
    # x = np.transpose(x, (1, 2))
    # x = np.transpose(x, (2, 3))
    # x = np.transpose(x, (3, 4))
    return x.transpose(1, 2).transpose(2, 3).transpose(3, 4)


def channel_first_3d(x):
    # x = np.transpose(x, (4, 3))
    # x = np.transpose(x, (3, 2))
    # x = np.transpose(x, (2, 1))
    return x.transpose(4, 3).transpose(3, 2).transpose(2, 1)


def load_reference_model(path):
    file = open(path)
    value_list = []
    scale = ''
    cnt = 0
    for line in file:
        if cnt == 0:
            scale = line
        elif cnt >= 3:
            line = line.replace('\n', '')
            value_list.append(float(line))
        cnt += 1
    scale_list = scale.split(' ')
    x = int(scale_list[0])
    y = int(scale_list[1])
    z = int(scale_list[2])
    ti = np.reshape(value_list, [z, y, x])
    for k in range(0, z):
        for i in range(0, y):
            for j in range(0, x):
                ti[k, i, j] = float(ti[k, i, j])
    return ti


def write_image_2_sgems_file(image, path):
    f = open(path, 'w')
    f.write(str(image.shape[1]) + ' ' + str(image.shape[0]) + ' ' + str(1) + '\n1\nv')
    for i in range(0, 1):
        for j in range(0, image.shape[1]):
            for k in range(0, image.shape[0]):
                value = image[j][k]
                f.write('\n' + str(value))
    f.close()


def write_model_2_sgems_file(image, path):
    f = open(path, 'w')
    f.write(str(image.shape[2]) + ' ' + str(image.shape[1]) + ' ' + str(image.shape[0]) + '\n1\nv')
    for i in range(0, image.shape[2]):
        for j in range(0, image.shape[1]):
            for k in range(0, image.shape[0]):
                value = image[k][j][i]
                f.write('\n' + str(value))
    f.close()


def write_image_2_vtk_file(image, path):
    vtk_header = '# vtk DataFile Version 3.4\nImage\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS '
    vtk_header += str(image.shape[0]) + ' ' + str(image.shape[1]) + ' ' + str(1) + '\n'
    vtk_header += 'ORIGIN 0.000000 0.000000 0.000000\n' + 'SPACING 0.100000 0.100000 0.100000\n'
    vtk_header += 'POINT_DATA ' + str(image.shape[1] * image.shape[0]) + '\n'
    vtk_header += 'SCALARS facies float 1\nLOOKUP_TABLE default'
    f = open(path, 'w')
    f.write(vtk_header)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            value = image[i][j]
            f.write('\n' + str(value))
    f.close()


def write_model_2_vtk_file(image, path):
    vtk_header = '# vtk DataFile Version 3.4\nImage\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS '
    vtk_header += str(image.shape[2]) + ' ' + str(image.shape[1]) + ' ' + str(image.shape[0]) + '\n'
    vtk_header += 'ORIGIN 0.000000 0.000000 0.000000\n' + 'SPACING 0.100000 0.100000 0.100000\n'
    vtk_header += 'POINT_DATA ' + str(image.shape[2] * image.shape[1] * image.shape[0]) + '\n'
    vtk_header += 'SCALARS facies float 1\nLOOKUP_TABLE default'
    f = open(path, 'w')
    f.write(vtk_header)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for k in range(0, image.shape[2]):
                value = image[i][j][k]
                f.write('\n' + str(value))
    f.close()


def get_file_name(dir_path):
    roots, dirs, files = [], [], []
    for root, dir, file in os.walk(dir_path):
        roots.append(root)
        dirs.append(dir)
        files.append(file)
    return roots, dirs, files


def convert_image_255(image):
    image = image * 255
    x, y = image.shape
    for i in range(0, x):
        for j in range(0, y):
            image[i][j] = math.floor(image[i][j])
    return image


# if __name__ == '__main__':
#     print(load_reference_model("/home/user/data/Cate_Hydro_3D/images/0.sgems"))
