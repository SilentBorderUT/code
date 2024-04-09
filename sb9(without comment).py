import pandas as pd
import concurrent.futures
import numpy as np
import dask.dataframe as dd

from concurrent.futures import ThreadPoolExecutor,wait
import scipy.interpolate as interp
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
import keyboard
import re
from sklearn.linear_model import RANSACRegressor
import itertools
from scipy.spatial import distance
import sys
import os
import plotly.io as pio
import argparse
import matplotlib.pyplot as plzt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.draw import line_nd
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import time
COLORMAP = 'viridis'
min_x = 0
max_x = 0
min_z = 0
max_z = 0
min_y = 0
max_y = 0
fig = None
stop_flag = False
import plotly.figure_factory as ff
import cv2
import imgkit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
voxel_size = 1
def line_intersection(x0, y0, z0, dx, dy, dz, min_x, max_x, min_y, max_y, min_z, max_z):
    t_min, t_max = None, None
    if dx != 0:
        t_min_x = (min_x - x0) / dx
        t_max_x = (max_x - x0) / dx
        if t_min_x < t_max_x:
            t_min, t_max = t_min_x, t_max_x
        else:
            t_min, t_max = t_max_x, t_min_x
    if dy != 0:
        t_min_y = (min_y - y0) / dy
        t_max_y = (max_y - y0) / dy
        if t_min is None or t_min_y < t_min:
            t_min = t_min_y
        if t_max is None or t_max_y > t_max:
            t_max = t_max_y
    if dz != 0:
        t_min_z = (min_z - z0) / dz
        t_max_z = (max_z - z0) / dz
        if t_min is None or t_min_z < t_min:
            t_min = t_min_z
        if t_max is None or t_max_z > t_max:
            t_max = t_max_z
    return t_min, t_max
def voxel_coordinates(x0, y0, z0, x1, y1, z1):
    coords = line_nd((x0, y0, z0), (x1, y1, z1))
    x_vals, y_vals, z_vals = coords
    return x_vals, y_vals, z_vals
def simulate_rays(grid, x_34, y_34, z_34, min_x, max_x, min_y, max_y, min_z, max_z):
    x0, y0, z0 = x_34 - min_x, y_34 - min_y, z_34 - min_z
    dx, dy, dz = np.random.uniform(-1, 1, 3)
    rays = [(x0, y0, z0, dx, dy, dz)]
    t_min, t_max = line_intersection(x0, y0, z0, dx, dy, dz, 0, max_x - min_x, 0, max_y - min_y, 0, max_z - min_z)
    if t_min is not None and t_max is not None:
        x1 = x0 + t_min * dx
        y1 = y0 + t_min * dy
        z1 = z0 + t_min * dz
        x2 = x0 + t_max * dx
        y2 = y0 + t_max * dy
        z2 = z0 + t_max * dz
        x_vals, y_vals, z_vals = voxel_coordinates(int(x1), int(y1), int(z1), int(x2), int(y2), int(z2))
        for x, y, z in zip(x_vals, y_vals, z_vals):
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and 0 <= z < grid.shape[2]:
                grid[y, x, z] += 1
    return grid
def save_slices(grid, title, event_id):
    save_folder = "voxel_slices"
    event_folder = os.path.join(save_folder, f"Event_{event_id}")
    if not os.path.exists(event_folder):
        os.makedirs(event_folder)
    for x in range(grid.shape[1]):
        x_slice = grid[:, x, :]
        plt.imshow(x_slice, cmap=COLORMAP, interpolation='nearest', origin='lower', vmin=np.min(grid), vmax=np.max(grid))
        plt.colorbar()
        plt.title(f"{title} - Event {event_id} - X Slice at X={x}")
        save_path = os.path.join(event_folder, f"x_slice_{x}.png")
        plt.savefig(save_path)
        plt.clf()
    for y in range(grid.shape[0]):
        y_slice = grid[y, :, :]
        plt.imshow(y_slice, cmap=COLORMAP, interpolation='nearest', origin='lower', vmin=np.min(grid), vmax=np.max(grid))
        plt.colorbar()
        plt.title(f"{title} - Event {event_id} - Y Slice at Y={y}")
        save_path = os.path.join(event_folder, f"y_slice_{y}.png")
        plt.savefig(save_path)
        plt.clf()
    for z in range(grid.shape[2]):
        z_slice = grid[:, :, z]
        plt.imshow(z_slice, cmap=COLORMAP, interpolation='nearest', origin='lower', vmin=np.min(grid), vmax=np.max(grid))
        plt.colorbar()
        plt.title(f"{title} - Event {event_id} - Z Slice at Z={z}")
        save_path = os.path.join(event_folder, f"z_slice_{z}.png")
        plt.savefig(save_path)
        plt.clf()
def save_figure(fig, filename):
    folder = "figures"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    pio.write_html(fig,file = filepath)
def angle_between_vectors(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
def stop_program():
    global stop_flag
    stop_flag = True
def bind_stop_key():
    keyboard.on_press_key('s', lambda _: stop_program())
def select_points_for_straight_line(data):
    z_values = data[:, 2]
    x_values = data[:, 0]
    y_values = data[:, 1]
    diff_indices = np.where((np.sqrt(np.diff(x_values)**2)+np.sqrt(np.diff(y_values)**2)+np.sqrt(np.diff(z_values)**2)) > 10)[0] + 1
    first_array = data[:diff_indices[0]]
    second_array = data[diff_indices[0]:diff_indices[1]]
    third_array = data[diff_indices[1]:]
    def fit_line_ransac(points):
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        model = RANSACRegressor()
        model.fit(X, y)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        return slope, intercept
    combinations = list(itertools.product(first_array, second_array, third_array))
    best_slope = None
    best_intercept = None
    best_residual = float('inf')
    for combination in combinations:
        points = np.array(combination)
        slope, intercept = fit_line_ransac(points)
        residual = np.sum(np.abs(slope * points[:, 0] + intercept - points[:, 1]))
        if residual < best_residual:
            best_slope = slope
            best_intercept = intercept
            best_residual = residual
    new_array = []
    for points in combination:
        new_array.append([points[0], best_slope * points[0] + best_intercept, points[2]])
    return (new_array)
def diff(line):
    x = line[:,0]
    y = line[:,1]
    z = line[:,2]
    distances = np.sqrt(np.diff(z)**2)+np.sqrt(np.diff(x)**2)+np.sqrt(np.diff(y)**2)
    return distances
def correct(array, n):
    count = 0
    for value in array:
        if value > n:
            count += 1
            if count > 1:
                return True
    return False
def select_incoming_outgoing(x_coords, y_coords, z_coords):
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2 + np.diff(z_coords)**2)
    change_index = np.argmax(distances)
    incoming_x = x_coords[:change_index+1]
    incoming_y = y_coords[:change_index+1]
    incoming_z = z_coords[:change_index+1]
    outgoing_x = x_coords[change_index+1:]
    outgoing_y = y_coords[change_index+1:]
    outgoing_z = z_coords[change_index+1:]
    return (incoming_x, incoming_y, incoming_z), (outgoing_x, outgoing_y, outgoing_z)
def closest_approach(line1, line2):
    def objective(s):
        p1 = np.array(line1[0]) + s * (np.array(line1[1]) - np.array(line1[0]))
        p2 = np.array(line2[0]) + s * (np.array(line2[1]) - np.array(line2[0]))
        return np.linalg.norm(p1 - p2)
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    s_star = result.x
    p1_star = np.array(line1[0]) + s_star * (np.array(line1[1]) - np.array(line1[0]))
    p2_star = np.array(line2[0]) + s_star * (np.array(line2[1]) - np.array(line2[0]))
    return p1_star, p2_star
def create_correct_path(incoming_line, outgoing_line):
    p1_star, p2_star = closest_approach(incoming_line, outgoing_line)
    point = (p1_star + p2_star) / 2
    control_points = [
        incoming_line[0],
        incoming_line[1],
        incoming_line[2],
        point,
        outgoing_line[0],
        outgoing_line[1],
        outgoing_line[2]
    ]
    return control_points, point,p1_star,p2_star
def calculate_min_distances(set1, set2):
    min_distances = []
    closest_indices = []
    for point1 in set1:
        distances = distance.cdist([point1], set2)
        min_distance = np.min(distances)
        closest_index = np.argmin(distances)
        if point1[0]< set2[closest_index][0]:
            x=-((point1[0]-set2[closest_index][0])**2)
        else:
            x=((point1[0]-set2[closest_index][0])**2)
        if point1[1]< set2[closest_index][1]:
            y=-((point1[1]-set2[closest_index][1])**2)
        else:
            y=((point1[1]-set2[closest_index][1])**2)
        if point1[2]< set2[closest_index][2]:
            z=-((point1[2]-set2[closest_index][2])**2)
        else:
            z=((point1[2]-set2[closest_index][2])**2)
        a = x+y+z
        if a < 0:
            min_distances.append(-min_distance)
        else :
            min_distances.append(min_distance)
        closest_indices.append(closest_index)
    average_point_error = np.mean(min_distances)
    return np.array(min_distances), np.array(closest_indices), average_point_error
def best_fitted(transport,points,weights,acceptable_error):
    initial_weights= [0.1,0.5,1.5,1.9,1]
    initial_errors = []  
    for weight in initial_weights:
        weights[3] = weight
        tck, u = interp.splprep(points.T, k=2, s=0, w=weights)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new, z_new = interp.splev(u_new, tck)
        fitted = np.array([x_new, y_new, z_new]).T
        _, _, avgg_error = calculate_min_distances(transport, fitted)
        initial_errors.append(abs(avgg_error))
    default_error = initial_errors[4]
    print("default error: ",default_error)
    sorted_pairs = sorted(zip(initial_weights, initial_errors), key=lambda pair: pair[1])
    sorted_weights = [pair[0] for pair in sorted_pairs]
    main_weight = sorted_weights[0]
    if main_weight< 1:
        direction = -1
    elif main_weight > 1:
        direction = 1
    elif main_weight == 1:
        return x_new,y_new,z_new,default_error,main_weight
    data_list = []
    while True:
        weights[3] += direction*0.05
        if 0>=weights[3] or weights[3]>=2:
            break
        tck, u = interp.splprep(points.T, k=2, s=0, w=weights)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new, z_new = interp.splev(u_new, tck)
        fittedd = np.array([x_new, y_new, z_new]).T
        _, _, avg_error = calculate_min_distances(transport, fittedd)
        data_list.append([abs(avg_error), x_new.tolist(), y_new.tolist(), z_new.tolist(),weights[3]])
        if abs(avg_error)< acceptable_error:
            break
    sorted_data_list = sorted(data_list, key=lambda x: x[0])
    smallest_avg_error = sorted_data_list[0]
    avg_error,x_new, y_new, z_new,the_weight = smallest_avg_error[0],smallest_avg_error[1], smallest_avg_error[2], smallest_avg_error[3], smallest_avg_error[4]
    print("accepte_error: ",avg_error)
    print(main_weight)
    return x_new,y_new,z_new,avg_error,the_weight
def extract_coordinates(track_data):
    x_coords = track_data["X(cm)"]
    y_coords = track_data["Y(cm)"]
    z_coords = track_data["Z(cm)"]
    hit_types = track_data["Volume"].apply(lambda x: "Incoming" if re.search(r'hodoscope\d+', x) else None)
    return x_coords, y_coords, z_coords, hit_types
def extract_transport_coordinates(track_data):
    xt_coords = track_data["X(cm)"]
    yt_coords = track_data["Y(cm)"]
    zt_coords = track_data["Z(cm)"]
    hit_types2 = track_data["Volume"].apply(lambda x: "Transport" if "VOI" in x else None)
    return xt_coords, yt_coords, zt_coords, hit_types2
def process_event(event_data,event_id, combined_grid):
    start_time = time.time()
    global min_x,max_x,min_y,min_z,max_z,max_y,fig
    unique_tracks = event_data["TrackID"].unique()
    for track_id in unique_tracks:
        track_data = event_data[event_data["TrackID"] == track_id]
        x_coords, y_coords, z_coords, hit_types = extract_coordinates(track_data)
        x_coords1, y_coords1, z_coords1, hit_types2 = extract_transport_coordinates(track_data)
        if len(x_coords) > 5:
            mask = hit_types.notnull()
            mask2 = hit_types2.notnull()
            if sum(mask) >= 6:
                (incoming_x, incoming_y, incoming_z), (outgoing_x, outgoing_y, outgoing_z) = select_incoming_outgoing(x_coords[mask], y_coords[mask], z_coords[mask])
                incoming_line = np.array([incoming_x, incoming_y, incoming_z]).T
                outgoing_line = np.array([outgoing_x, outgoing_y, outgoing_z]).T
                if len(incoming_line) >= 3 and len(outgoing_line) >= 3 and incoming_line[-1][2] > outgoing_line[0][2]:
                    diffin = correct(diff(incoming_line),10)
                    diffout = correct(diff(outgoing_line),10)
                    if diffin== True and diffout == True:
                        incoming_line = select_points_for_straight_line(incoming_line)
                        outgoing_line = select_points_for_straight_line(outgoing_line)               
                        control_points, point,poca_in,poca_out = create_correct_path(incoming_line, outgoing_line)
                        x_coords = x_coords[mask]
                        y_coords = y_coords[mask]
                        z_coords = z_coords[mask]
                        hit_types = hit_types[mask]                        
                        x_coord = [v[0] for v in control_points]
                        y_coord = [v[1] for v in control_points]
                        z_coord = [v[2] for v in control_points]
                        incoming_x_2 = [x_coord[1], x_coord[2]]
                        incoming_y_2 = [y_coord[1], y_coord[2]]
                        incoming_z_2 = [z_coord[1], z_coord[2]]
                        x_34 = [x_coord[2], x_coord[3]]
                        y_34 = [y_coord[2], y_coord[3]]
                        z_34 = [z_coord[2], z_coord[3]]
                        x_45 = [x_coord[3], x_coord[4]]
                        y_45 = [y_coord[3], y_coord[4]]
                        z_45 = [z_coord[3], z_coord[4]]
                        x_23_45 = [x_coord[1], x_coord[2], x_coord[3], x_coord[4]]
                        y_23_45 = [y_coord[1], y_coord[2], y_coord[3], y_coord[4]]
                        z_23_45 = [z_coord[1], z_coord[2], z_coord[3], z_coord[4]]
                        x_23_34_45 = [x_coord[1], x_coord[2], x_coord[2], x_coord[3],  x_coord[3], x_coord[4]]
                        y_23_34_45 = [y_coord[1], y_coord[2], y_coord[2], y_coord[3],  y_coord[3], y_coord[4]]
                        z_23_34_45 = [z_coord[1], z_coord[2], z_coord[2], z_coord[3],  z_coord[3], z_coord[4]]
                        points = np.array([x_coord, y_coord, z_coord]).T
                        unique_points = np.unique(points, axis=0)
                        weights = np.ones(len(unique_points))
                        x_transport = x_coords1[mask2]
                        y_transport = y_coords1[mask2]
                        z_transport = z_coords1[mask2]
                        transport = np.array([x_transport, y_transport, z_transport]).T
                        a_34 = np.array([x_34, y_34, z_34]).T

                        x1_in, y1_in, z1_in = control_points[0]
                        x2_in, y2_in, z2_in = control_points[1]
                        x3_in, y3_in, z3_in = control_points[2]
                        x_poca, y_poca, z_poca = control_points[3]
                        x4_out, y4_out, z4_out = control_points[4]
                        x5_out, y5_out, z5_out = control_points[5]
                        x6_out, y6_out, z6_out = control_points[6]
                        angle_123 = angle_between_vectors([x2_in - x1_in, y2_in - y1_in, z2_in - z1_in], [x3_in - x2_in, y3_in - y2_in, z3_in - z2_in])
                        angle_234 = angle_between_vectors([x3_in - x2_in, y3_in - y2_in, z3_in - z2_in], [x4_out - x3_in, y4_out - y3_in, z4_out - z3_in])
                        angle_345 = angle_between_vectors([x4_out - x3_in, y4_out - y3_in, z4_out - z3_in], [x5_out - x4_out, y5_out - y4_out, z5_out - z4_out])
                        angle_456 = angle_between_vectors([x5_out - x4_out, y5_out - y4_out, z5_out - z4_out], [x6_out - x5_out, y6_out - y5_out, z6_out - z5_out])
                        angle_3poca4 = angle_between_vectors([x_poca - x3_in, y_poca - y3_in, z_poca - z3_in], [x4_out - x_poca, y4_out - y_poca, z4_out - z_poca])
                        if angle_234 > 1:
                            break
                        print(angle_234)
                        x_34 = np.array(x_poca)
                        y_34 = np.array(y_poca)
                        z_34 = np.array(z_poca)
                        voxel_size = 1
                        density_grid = np.zeros((int((max_y - min_y) // voxel_size), int((max_x - min_x) // voxel_size), int((max_z - min_z) // voxel_size)), dtype=int)
                        grid = simulate_rays(density_grid, x_34,y_34,z_34, min_x, max_x, min_y, max_y, min_z, max_z)
                        combined_grid +=grid
                        end_time = time.time()
                        elapse = end_time - start_time

                        print("time: ", elapse)
                        

def read_csv_file_concurrent(filename, chunk_size=None):
    if chunk_size:
        chunks = dd.read_csv(filename, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                             names=["EventID", "X(cm)", "Y(cm)", "Z(cm)", "GlobalTime(ns)", "TrackID", "Particle(PDG)", "Volume", "Process_name"],
                             assume_missing=True, dtype={'EventID': 'float64'}, blocksize=chunk_size)
        return chunks.compute()
    else:
        return pd.read_csv(filename, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                           names=["EventID", "X(cm)", "Y(cm)", "Z(cm)", "GlobalTime(ns)", "TrackID", "Particle(PDG)", "Volume", "Process_name"],
                           low_memory=False, engine='c')

def read_csv_concurrent(filename, chunk_size=None):
    with ThreadPoolExecutor() as executor:
        if chunk_size:
            future = executor.submit(read_csv_file_concurrent, filename, chunk_size)
        else:
            future = executor.submit(read_csv_file_concurrent, filename)
        return future.result()

def sort_data(data):
    data.sort_values(by=["EventID", "TrackID"], inplace=True)
    return data
parser = argparse.ArgumentParser(description="Process simulation data and generate output files.")
parser.add_argument("input_file", type=str, help="Path to the input CSV file")
parser.add_argument("--plot", action="store_true", help="Visualize the results")
args = parser.parse_args()
filename = args.input_file
chunk_size = 62318749
start_tim=time.time()
data = read_csv_concurrent(filename, chunk_size=chunk_size)
end=time.time()
elapsee = end - start_tim

print("time: ", elapsee)
data = sort_data(data)

unique_events = data["EventID"].unique()

parser = argparse.ArgumentParser(description="Process simulation data and generate output files.")
parser.add_argument("input_file", type=str, help="Path to the input CSV file")
parser.add_argument("--plot", action="store_true", help="Visualize the results")
def process_chunk(chunk_of_events, data,combined_grid):
    global stop_flag, fig

    for event_id in chunk_of_events:
        if stop_flag:
            break
        event_data = data[data["EventID"] == event_id]
        process_event(event_data, event_id, combined_grid)


def main():
    global min_x,max_x,min_y,min_z,max_z,max_y,fig
    x_coor, y_coor, z_coor, hit_typ = extract_coordinates(data)
    masks = hit_typ.notnull()
    x_coor = x_coor[masks]
    y_coor = y_coor[masks]
    z_coor = z_coor[masks]
    fig = go.Figure()
    if sum(masks) >= 6:
        min_x = min(x_coor) 
        max_x = max(x_coor) 
        min_y = min(y_coor)
        max_y = max(y_coor)
        min_z = min(z_coor) 
        max_z = max(z_coor)
        min_x_det = min_x 
        max_x_det = max_x
        min_y_det = min_y
        max_y_det = max_y
        min_z_det = min_z
        max_z_det = max_z
        side1_corners = [
            [min_x_det, min_y_det, min_z_det],
            [max_x_det, min_y_det, min_z_det],
            [max_x_det, max_y_det, min_z_det],
            [min_x_det, max_y_det, min_z_det]
        ]
        side2_corners = [
            [min_x_det, min_y_det, max_z_det],
            [max_x_det, min_y_det, max_z_det],
            [max_x_det, max_y_det, max_z_det],
            [min_x_det, max_y_det, max_z_det]
        ]
        side3_corners = [
            [min_x_det, min_y_det, min_z_det],
            [min_x_det, max_y_det, min_z_det],
            [min_x_det, max_y_det, max_z_det],
            [min_x_det, min_y_det, max_z_det]
        ]
        side4_corners = [
            [max_x_det, min_y_det, min_z_det],
            [max_x_det, max_y_det, min_z_det],
            [max_x_det, max_y_det, max_z_det],
            [max_x_det, min_y_det, max_z_det]
        ]
        side5_corners = [
            [min_x_det, min_y_det, min_z_det],
            [max_x_det, min_y_det, min_z_det],
            [max_x_det, min_y_det, max_z_det],
            [min_x_det, min_y_det, max_z_det]
        ]
        side6_corners = [
            [min_x_det, max_y_det, min_z_det],
            [max_x_det, max_y_det, min_z_det],
            [max_x_det, max_y_det, max_z_det],
            [min_x_det, max_y_det, max_z_det]
        ]
        side1_frame = go.Scatter3d(
            x=[corner[0] for corner in side1_corners + [side1_corners[0]]],
            y=[corner[1] for corner in side1_corners + [side1_corners[0]]],
            z=[corner[2] for corner in side1_corners + [side1_corners[0]]],
            mode='lines',
            line=dict(color='black', width=2)
        )
        side2_frame = go.Scatter3d(
            x=[corner[0] for corner in side2_corners + [side2_corners[0]]],
            y=[corner[1] for corner in side2_corners + [side2_corners[0]]],
            z=[corner[2] for corner in side2_corners + [side2_corners[0]]],
            mode='lines',
            line=dict(color='black', width=2)
        )
        side3_frame = go.Scatter3d(
            x=[corner[0] for corner in side3_corners + [side3_corners[0]]],
            y=[corner[1] for corner in side3_corners + [side3_corners[0]]],
            z=[corner[2] for corner in side3_corners + [side3_corners[0]]],
            mode='lines',
            line=dict(color='black', width=2)
        )
        side4_frame = go.Scatter3d(
            x=[corner[0] for corner in side4_corners + [side4_corners[0]]],
            y=[corner[1] for corner in side4_corners + [side4_corners[0]]],
            z=[corner[2] for corner in side4_corners + [side4_corners[0]]],
            mode='lines',
            line=dict(color='black', width=2)
        )
        side5_frame = go.Scatter3d(
            x=[corner[0] for corner in side5_corners + [side5_corners[0]]],
            y=[corner[1] for corner in side5_corners + [side5_corners[0]]],
            z=[corner[2] for corner in side5_corners + [side5_corners[0]]],
            mode='lines',
            line=dict(color='black', width=2)
        )
        side6_frame = go.Scatter3d(
            x=[corner[0] for corner in side6_corners + [side6_corners[0]]],
            y=[corner[1] for corner in side6_corners + [side6_corners[0]]],
            z=[corner[2] for corner in side6_corners + [side6_corners[0]]],
            mode='lines',
            line=dict(color='black', width=2)
        )
        fig.add_trace(side1_frame)
        fig.add_trace(side2_frame)
        fig.add_trace(side3_frame)
        fig.add_trace(side4_frame)
        fig.add_trace(side5_frame)
        fig.add_trace(side6_frame)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[min_x_det - 0, max_x_det + 0]),
                yaxis=dict(range=[min_y_det - 0, max_y_det + 0]),
                zaxis=dict(range=[min_z_det - 0, max_z_det + 0])
            )
        )
    print("done")
    combined_grid = np.zeros((int((max_y - min_y) // voxel_size), int((max_x - min_x) // voxel_size), int((max_z - min_z) // voxel_size)), dtype=int)
    unique_events_chunk_size = 50000# Adjust the chunk size as needed for unique_events
    print(len(unique_events))
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []

        # Split unique_events into chunks and submit each chunk for parallel processing
        for chunk_start in range(0, len(unique_events), unique_events_chunk_size):
            chunk_of_events = unique_events[chunk_start:chunk_start + unique_events_chunk_size]
            future = executor.submit(process_chunk, chunk_of_events, data,combined_grid)
            futures.append(future)
        wait(futures)

    
    title = "Voxel Visualization (Combined)"
    a = 1
    save_slices(combined_grid, title,a)
if __name__ == "__main__":
    bind_stop_key()
    
    main()
