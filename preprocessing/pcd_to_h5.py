import open3d as o3d
import os
import numpy as np
import h5py


DATASET = "new_test"

SUP_DIR = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname( __file__ ))))
DATA_DIR = os.path.abspath(os.path.join(SUP_DIR, 'data'))
DATA_SET_DIR = os.path.abspath(os.path.join(DATA_DIR, DATASET))
RAW_SCANS = os.path.abspath(os.path.join(DATA_SET_DIR, 'raw_scans'))


# def downsample(pcd):
#     print("Downsample the point cloud with a voxel of 0.005")
#     downpcd = pcd.voxel_down_sample(voxel_size=0.005)
#     o3d.visualization.draw_geometries([downpcd])

def convert_scans():
    try:
        os.mkdir(os.path.join(DATA_SET_DIR, "h5_files"))
    except:
        pass
    scans = os.listdir(RAW_SCANS)
    for scan in scans:
        full_scan = o3d.io.read_point_cloud(os.path.join(RAW_SCANS, scan, 'PointCloudCapture.pcd'))
        head = o3d.io.read_point_cloud(os.path.join(RAW_SCANS, scan, 'face_segment.pcd'))
        print(full_scan)
        print(head)
        data = np.asarray(full_scan.points, dtype='f4')[:]
        head_data = np.asarray(head.points, dtype='f4')
        data = data.tolist()
        data = [tuple(point) for point in data]
        head_data = set([tuple(point) for point in head_data])
        labels = [point in head_data for point in data]

        #Verify all the points got selected
        head_selected = np.array(data)[labels]
        if len(head_selected) != len(head_data):
            print(f"\n\nNot all head points were labeled! ({len(head_selected)}/{len(head_data)})\n\n")
        # print(np.asarray(pcd.points))
        # o3d.visualization.draw_geometries([pcd])

        with h5py.File(os.path.join(DATA_SET_DIR, "h5_files", f'{scan}.h5'), "w") as f:
            f['data'] = data
            f['labels'] = labels


if __name__ == "__main__":
    convert_scans()
