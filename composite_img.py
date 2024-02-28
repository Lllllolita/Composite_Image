from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.spatial import KDTree
import math
import random
# np.random.seed(0)
# random.seed(0)
def is_not_overlapped(tar_min_X, tar_max_X, tar_min_Y, tar_max_Y, d_min_X, d_max_X, d_min_Y, d_max_Y):
    if tar_max_X < d_min_X or tar_min_X > d_max_X or tar_max_Y < d_min_Y or tar_min_Y > d_max_Y:
        return True
    else:
        return False
def get_crop_fg(mask, img):
    n_mask = np.array(mask)
    points = np.where(n_mask == 255)
    x1 = np.min(points[1])
    x2 = np.max(points[1])
    y1 = np.min(points[0])
    y2 = np.max(points[0])
    return mask.crop((x1, y1, x2, y2)), img.crop((x1, y1, x2, y2))

bg_img_path = "Images\\000849.jpg"
fg_img_path = "fg\\fg.jpg"
mask_img_path = "fg\\mask.jpg"
box_path = "Annotations\\000849.jpg.txt"

bg_img = Image.open(bg_img_path)
width, height = bg_img.size

boxes = []
with open(box_path) as f:
    lines = f.readlines()[1:]
    for line in lines:
        boxes.append(list(map(int, line.split())))
for i, point in enumerate(boxes):
    score = point[0]
    x1, y1, x2, y2 = point[1:]
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    h = y2 - y1
    boxes[i] = [score, x1, x2, y1, y2, h, x_center, y_center]

box_arr = np.array(boxes)
X = box_arr[:, 4:6]
model = AffinityPropagation(damping=0.5)
model.fit(X)

yhat = model.predict(X)
clusters = np.unique(yhat)
h_arr = []
cluster_bg_img = bg_img.copy()
draw = ImageDraw.Draw(cluster_bg_img)
base_color = 0
color_num = len(clusters)
color_bar = int(255 / color_num)
count = 0
for cluster in clusters:
    iter_idx = np.where(yhat == cluster)
    iter_data = np.take(box_arr, iter_idx, axis=0)[0]
    draw_x1, draw_x2 = np.min(iter_data[:, 1]), np.max(iter_data[:, 2])
    draw_y1, draw_y2 = np.min(iter_data[:, 3]), np.max(iter_data[:, 4])
    outline_color = (0, 255 - (base_color + count * color_bar), 0)
    count += 1
    draw.rectangle([(draw_x1, draw_y1), (draw_x2, draw_y2)], fill=None, outline=outline_color, width=3)
    mean_h = np.mean(iter_data[:, 5])
    h_arr.append(mean_h)
cluster_bg_img.show()
h_arr = np.array(h_arr)
sorted_h_arr = np.argsort(-h_arr)
print(sorted_h_arr)
_eps = 5
cycle = True
while(cycle):
    tar_cluster = np.random.choice(clusters)
    print(tar_cluster)
    tar_points_idx = np.where(yhat == tar_cluster)
    tar_points = np.take(box_arr, tar_points_idx, axis=0)[0]
    print(tar_points)
    if tar_cluster == sorted_h_arr[0]:
        print("stage1")
        cycle = False
        tar_point_idx = np.argmax(tar_points[:, 5])
        tar_point = tar_points[tar_point_idx]
        res_h = int(tar_point[5] * random.uniform(1, 1.1))
        res_x = random.randint(min(0, tar_point[6] - int(width * 0.3)), max(width, tar_point[6] + int(width * 0.3)))
        res_y = int(tar_point[7] + random.uniform(0, res_h * 0.1))
    else:
        other_clusters_idx = sorted_h_arr[: np.argwhere(tar_cluster == sorted_h_arr)[0][0]]
        other_cluster_data = []
        for i in other_clusters_idx:
            data_idx = np.where(yhat == i)
            other_cluster_data.append(np.take(box_arr, data_idx, axis=0)[0])
        overlap_clusters_idx = []
        tar_min_X, tar_max_X = np.min(tar_points[:, 1]), np.max(tar_points[:, 2])
        tar_min_Y, tar_max_Y = np.min(tar_points[:, 3]), np.max(tar_points[:, 4])
        for i, d in enumerate(other_cluster_data):
            d_min_X, d_max_X = np.min(d[:, 1]), np.max(d[:, 2])
            d_min_Y, d_max_Y = np.min(d[:, 3]), np.max(d[:, 4])
            if is_not_overlapped(tar_min_X, tar_max_X, tar_min_Y, tar_max_Y, d_min_X, d_max_X, d_min_Y, d_max_Y):
                pass
            else:
                overlap_clusters_idx.append(i)
        if len(overlap_clusters_idx) == 0:
            cycle = False
            tar_point_idx = np.argmax(tar_points[:, 5])
            tar_point = tar_points[tar_point_idx]
            res_h = int(tar_point[5] * random.uniform(1, 1.1))
            res_x = random.randint(np.min(tar_points[:, 1]), np.max(tar_points[:, 2]))
            res_y = int(tar_point[7] + random.uniform(0, res_h * 0.1))
        else:
            no_lapped_points = []
            min_x_range = dict()
            for i in overlap_clusters_idx:
                tree = KDTree(other_cluster_data[i][:, 6:])
                pairs = []
                k_num = math.ceil(other_cluster_data[i].shape[0] / 2)
                k_sequence = [i + 1for i in range(k_num)]
                cur_no_lapped_points = []
                for j, point in enumerate(tar_points):
                    flag = True
                    distances, indices = tree.query(point[6:], k=k_sequence, p=1)
                    p_x1, p_x2 = point[1], point[2]
                    p_y1, p_y2 = point[3], point[4]
                    for k in indices:
                        other_point = other_cluster_data[i][k]
                        o_x1, o_x2 = other_point[1], other_point[2]
                        o_y1, o_y2 = other_point[3], other_point[4]
                        if is_not_overlapped(p_x1, p_x2, p_y1, p_y2, o_x1, o_x2, o_y1, o_y2):
                            pass
                        else:
                            flag = False
                            break
                    if flag:
                        cur_no_lapped_points.append(j)
                        right_x1_idx = np.where(p_x2 < other_cluster_data[i][:, 1])
                        left_x2_idx = np.where(p_x1 > other_cluster_data[i][:, 2])
                        right_x1_list = np.take(other_cluster_data[i][:, 1], right_x1_idx)[0]
                        left_x2_list = np.take(other_cluster_data[i][:, 2], left_x2_idx)[0]
                        if len(right_x1_list) > 0 and len(left_x2_list) > 0:
                            if str(j) not in min_x_range.keys():       
                                min_x_range[str(j)] = [np.max(left_x2_list), np.min(right_x1_list)]
                            else:
                                min_x_range[str(j)][0] = max(min_x_range[str(j)][0], np.max(left_x2_list))
                                min_x_range[str(j)][1] = min(min_x_range[str(j)][1], np.min(right_x1_list))
                        elif len(right_x1_list) > 0 and len(left_x2_list) == 0:
                            if str(j) not in min_x_range.keys():       
                                min_x_range[str(j)] = [np.min(tar_points[:, 1]), np.min(right_x1_list)]
                            else:
                                min_x_range[str(j)][0] = max(min_x_range[str(j)][0], np.min(tar_points[:, 1]))
                                min_x_range[str(j)][1] = min(min_x_range[str(j)][1], np.min(right_x1_list))
                        elif len(right_x1_list) == 0 and len(left_x2_list) > 0:
                            if str(j) not in min_x_range.keys():
                                min_x_range[str(j)] = [np.max(left_x2_list), np.max(tar_points[:, 2])]
                            else:
                                min_x_range[str(j)][0] = max(min_x_range[str(j)][0], np.max(left_x2_list))
                                min_x_range[str(j)][1] = min(min_x_range[str(j)][1], np.max(tar_points[:, 2]))
                        else:
                            if str(j) not in min_x_range.keys():
                                min_x_range[str(j)] = [np.min(tar_points[:, 1]), np.max(tar_points[:, 2])]
                            else:
                                min_x_range[str(j)][0] = max(min_x_range[str(j)][0], np.min(tar_points[:, 1]))
                                min_x_range[str(j)][1] = min(min_x_range[str(j)][1], np.max(tar_points[:, 2]))
                if len(no_lapped_points) == 0:
                    no_lapped_points = cur_no_lapped_points
                else:
                    no_lapped_points = list(set(no_lapped_points) & set(cur_no_lapped_points))
            if len(no_lapped_points) > 0:
                cycle = False
                tar_point_idx = random.choice(no_lapped_points)
                tar_point = tar_points[tar_point_idx]
                res_h = int(tar_point[5] * random.uniform(1, 1.1))
                res_y = min(int(tar_point[7] + random.uniform(0, res_h * 0.1)), np.max(tar_points[:, 7]))
                res_x = random.randint(min_x_range[str(tar_point_idx)][0] + int(res_h * 0.2), min_x_range[str(tar_point_idx)][1] - int(res_h * 0.2))
            else:
                clusters = np.delete(clusters, np.where(clusters == tar_cluster))

fg_img = Image.open(fg_img_path)
fg_mask = Image.open(mask_img_path).convert('L')
fg_mask, fg_img = get_crop_fg(fg_mask, fg_img)
width, height = fg_img.size
scale = res_h / height
res_size = (int(scale * width), int(scale * height))
fg_img = fg_img.resize(res_size)
fg_mask = fg_mask.resize(res_size)
point = (max(0, int(res_x - res_size[0] // 2)), max(0, int(res_y - res_size[1] // 2)))
bg_img.paste(fg_img, point, mask=fg_mask)
bg_img.show()