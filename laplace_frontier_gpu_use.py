# Now doing the edge/frontier thing
import numpy as np
import math
import random
from PIL import Image
import cv2
import imageio
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
from rtree import index
import cProfile
import torch

log_file = open('output.log', 'w')
sys.stdout = log_file
count = 0

class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = -1
    def equals(self, other):
        return self.x == other.x and self.y == other.y
def distance(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
def smoothness(nodes):
        s = 0.0
        if len(nodes) > 2:
            a = distance(nodes[0], nodes[1])
            line_segs = 0
            for i in range(2, len(nodes)):
                b = distance(nodes[i - 1], nodes[i])
                c = distance(nodes[i - 2], nodes[i])
                if a != 0 and b != 0:
                    acosValue = (a * a + b * b - c * c) / (2.0 * a * b)
                    if -1.0 < acosValue < 1.0:
                        angle = math.pi - math.acos(acosValue)
                        k = 2.0 * angle / (a + b)
                        s += k * k
                        line_segs += 1
                a = b
        return s / line_segs


timesarray = []
testarray = []
testbranchesarray = []
teststepsizearray = []
testlaplaceperarray = []
testimagearray = []
laplacetimearray = []
randompointtimearray = []
graddescenttimearray = []
smoothnessarray = []

def LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always, node_list_shape, indices, counter, points_to_check, counters_set_points):
    # CPU with Numpy array (requires kernel to be numpy as well)
    # potential_map = cv2.filter2D(potential_map, -1, kernel)
    # GPU with Cupy array (requires kernel to be cupy as well)
    potential_map = torch.nn.functional.conv2d(potential_map, kernel, padding=1)

    potential_map[0][0][end[1]][end[0]] = 0
    potential_map[0][0][node_list_shape] = 0
    potential_map[0][0][indices] = counter
    potential_map[0][0][points_to_check[:, 0], points_to_check[:, 1]] = counters_set_points
    return potential_map

# RRT Algorithm
def RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion, branches_before_each_laplace, visualization, output_path):
    height = len(image)
    length = len(image[0])
    this_node = Node(end[0], end[1])
    node_list.append(this_node)

    total_iter = 0
    i = 1
    pathFound = False

    potential_map = np.ones((height, length))
    node_list_shape = np.zeros((height, length), dtype=bool)
    nodes_array = index.Index()
    nodes_array.insert(1, (this_node.x, this_node.y, this_node.x, this_node.y), obj=this_node)

    potential_map[end[1]][end[0]] = 0
    boundary_x = []
    boundary_y = []
    for y in range(height):
        for x in range(length):
            if(image[y][x][0] == 0):
                boundary_x.append(x)
                boundary_y.append(y)
    kernel = torch.tensor([[0, 0.25, 0],
                       [0.25, 0, 0.25],
                       [0, 0.25, 0]], dtype=torch.double).unsqueeze(0).unsqueeze(0).to(device="cuda")

    start_time = time.time()
    height_always = potential_map.shape[0]
    length_always = potential_map.shape[1]

    # For GPU, must convert to cupy
    potential_map = torch.from_numpy(potential_map)
    potential_map = potential_map.unsqueeze(0).unsqueeze(0).to(device="cuda")
    indices = np.zeros((height, length), dtype=bool)
    indices[boundary_y, boundary_x] = True
    indices[:, [0, -1]] = True
    indices[[0, -1], :] = True



    # map_counter_to_points = {}
    # map_counter_to_points[0] = [(end[1],end[0])]
    points_to_check = np.array([(end[1],end[0])])
    set_of_points = set()
    set_of_points.add((end[1],end[0]))
    counters_set_points = torch.from_numpy(np.array([0], dtype=np.float64)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float64, device="cuda")



    potential_map = potential_map.squeeze(0).squeeze(0)
    potential_map = potential_map.cpu().numpy()
    counter = 1
    while potential_map[start[1],start[0]] == counter:#(potential_map[start[1],start[0]] <= 0.1 + counter and potential_map[start[1],start[0]] >= counter - 0.1) or (potential_map[start[1],start[0]] <= 0.1 + counter - 1 and potential_map[start[1],start[0]] >= counter - 1 - 0.1):#potential_map[start[1],start[0]] == counter or potential_map[start[1],start[0]] == counter - 1:
      potential_map = torch.from_numpy(potential_map)
      potential_map = potential_map.unsqueeze(0).unsqueeze(0).to(device="cuda")
      mask = [potential_map == counter]
      for _ in range(2000): # laPlace iters - 500 seems to be good
        potential_map[mask] = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always,node_list_shape, indices, counter, points_to_check, counters_set_points)[mask]
      potential_map = potential_map.squeeze(0).squeeze(0)
      potential_map = potential_map.cpu().numpy()
      potential_map_as_image = np.uint8(255 / counter* potential_map)
      potential_map_as_image[(potential_map_as_image == 255 ) | (potential_map_as_image < 254)] = 0
      potential_map_as_image[potential_map_as_image != 0] = 255 
      edge = cv2.Canny(potential_map_as_image, 50, 150)
      edge_points = np.argwhere(edge > 0)
      # print(len(edge_points))
      potential_map[potential_map == counter] = counter + 1
      for edge_point in edge_points:
        if (edge_point[0], edge_point[1]) not in set_of_points:
          np.append(points_to_check,(edge_point[0], edge_point[1]))#points_to_check.append((edge_point[0], edge_point[1]))
          # counters_set_points.append(counter)
          new_data = torch.from_numpy(np.array([counter], dtype=np.float64)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float64, device="cuda")
          tensor = torch.cat((counters_set_points, new_data))
          set_of_points.add((edge_point[0], edge_point[1]))
      # potential_map[edge_points[:, 0], edge_points[:, 1]] = counter
      im = Image.open(file_name)
      counter = counter + 1
      result = im.copy()
      height = len(image)
      length = len(image[0])
      draw_result(image, result, start, end, node_list, potential_map=np.zeros((height, length)), show_expansion=False)
      for x in range(0, length):
              for y in range(0, height):
                  if potential_map[y][x] == counter:
                      toPut =  0
                      result.putpixel((x, y), (toPut, toPut, toPut))
                  else:
                      toPut = 250 - int(potential_map[y][x]*225/counter)
                      result.putpixel((x, y), (toPut, toPut, toPut))
      result.save(str(counter) + '.png')
      # print("count", counter)
      # print(potential_map[start[1],start[0]])
      
    test = try_grad_descent(potential_map, step_size, start[0], start[1], node_list, nodes_array)
    # print("test", len(test))
    # print(counter)
    # print(potential_map[start[1],start[0]])
    im = Image.open(file_name)
    result = im.copy()
    node_list.extend(test)
    nearest_nodes = list(nodes_array.intersection((start[0] - 1, start[1] - 1, start[0] + 1, start[1] + 1), objects=True)) # instead of 3, using 1
    #
    result = im.copy()
    height = len(image)
    length = len(image[0])
    draw_result(image, result, start, end, node_list, potential_map=np.zeros((height, length)), show_expansion=False)
    for x in range(0, length):
            for y in range(0, height):
                if potential_map[y][x] == counter:
                    toPut =  0
                    result.putpixel((x, y), (toPut, toPut, toPut))
                else:
                    toPut = 250 - int(potential_map[y][x]*225/counter)
                    result.putpixel((x, y), (toPut, toPut, toPut))
    result.save(output_path + '_found_path_smoothed.png')

    #

    current_node = nearest_nodes[0].object
    found_path = []
    while current_node.parent != -1:
        found_path.append(current_node)
        current_node = node_list[current_node.parent]
    new_smooth = smoothness(found_path)
    print("Smoothness new metric from matlab", new_smooth)
    smoothnessarray.append(new_smooth)

    if(len(found_path) > 1):
        # print("Len found path:", len(found_path))
        x = np.array([node.x for node in found_path])
        y = np.array([node.y for node in found_path])

        dx = np.gradient(x)
        dy = np.gradient(y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        squared_second_derivative = ddx**2 + ddy**2

        for i in range(len(found_path)):
            node = found_path[i]
            x = math.floor(node.x)
            y = math.floor(node.y)
            # for x in range(round_x - 1, round_x + 1):
            #     for y in range(round_y - 1, round_y + 1):
            #         if(0 < x and x < length - 1 and 0 < y and y < height - 1):
            # if result.getpixel((x, y))[1] != 0 and result.getpixel((x, y))[1] != 255:
            #     print("Duplicate")
            result.putpixel((x, y), (0, 20 + int(squared_second_derivative[i] * 235 / max(squared_second_derivative)), 0))
            # print(int(squared_second_derivative[i] * 235 / max(squared_second_derivative)))


        result.save(output_path + '_found_path.png')
        # smooth(found_path, potential_map, boundary conditions?, junction points?, )
    else:
        print("Proper solution not found")
    return node_list, found_path, potential_map, indices


      
def try_grad_descent(potential_map, step_size, new_x, new_y, node_list, nodes_array, an_obj=None):


    def distance(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def smoothness(nodes):
        s = 0.0
        if len(nodes) > 2:
            a = distance(nodes[0], nodes[1])
            line_segs = 0
            for i in range(2, len(nodes)):
                b = distance(nodes[i - 1], nodes[i])
                c = distance(nodes[i - 2], nodes[i])
                if a != 0 and b != 0:
                    acosValue = (a * a + b * b - c * c) / (2.0 * a * b)
                    if -1.0 < acosValue < 1.0:
                        angle = math.pi - math.acos(acosValue)
                        k = 2.0 * angle / (a + b)
                        s += k * k
                        line_segs += 1
                a = b
        if line_segs == 0:
            return 20
        return s / line_segs



    poser = round(new_y)
    posec = round(new_x)
    toAppend = Node(posec, poser)

    limit = 0
    new_node_list = []
    new_node_list.append(toAppend)
    index_curr = len(node_list) + 1
    while True:
        if limit > 1000:
            return []

        if 0 < poser < potential_map.shape[0]-1 and 0 < posec < potential_map.shape[1]-1:
            gradr = potential_map[int(poser+1)][int(posec)] - potential_map[int(poser-1)][int(posec)]
            gradc = potential_map[int(poser)][int(posec+1)] - potential_map[int(poser)][int(posec-1)]
        else:
            return []
        maggrad = math.sqrt(gradr**2 + gradc**2)

        if(maggrad != 0):
            a = step_size / maggrad
            poser = poser-a*gradr # round?
            posec = posec-a*gradc # round?
        # poser = poser-10000*gradr
        # posec = posec-10000*gradc
        # print(poser, posec)
        toAppend = Node(posec, poser)
        new_node_list[-1].parent = index_curr
        index_curr += 1
        new_node_list.append(toAppend)
        limit = limit + 1
        nearest_nodes = None
        if 0 < posec < potential_map.shape[1]-1 and 0 < poser < potential_map.shape[0]-1:
            nearest_nodes = list(nodes_array.intersection((posec - 1 , poser - 1, posec + 1, poser + 1), objects=True)) # using 1 instead of 3
        valid = []
        if nearest_nodes:
            for node_found in nearest_nodes:
                if an_obj is None or node_found.object != an_obj:
                  valid.append(node_found.object)
        if len(valid) != 0:
            # print("End of Path")
            node_to_smoothness = []
            for node_in_valid in valid:
                lst = []
                lst.append(new_node_list[-2])
                lst.append(new_node_list[-1])
                lst.append(node_in_valid)
                if(node_in_valid.parent) and an_obj == None:
                    lst.append(node_list[node_in_valid.parent])
                node_to_smoothness.append([node_in_valid, smoothness(lst)])
            least_smooth = node_to_smoothness[0]
            for every in node_to_smoothness:
                if least_smooth[1] > every[1]:
                    least_smooth = every
            for i, object in enumerate(node_list):
                if object.equals(least_smooth[0]):
                    # ADD
                    # two objects are: new_node_list[-1], and object
                    # start_connect_x = new_node_list[-1].x
                    # start_connect_y = new_node_list[-1].y
                    # end_connect_x = object.x
                    # end_connect_y = object.y
                    # diff_x = end_connect_x - start_connect_x
                    # diff_y = end_connect_y - start_connect_y
                    # new_node_list[-1].parent = index_curr
                    # index_curr += 1
                    # new_node_list.append(Node(start_connect_x + diff_x/4, start_connect_y + diff_y/4))
                    # new_node_list[-1].parent = index_curr
                    # index_curr += 1
                    # new_node_list.append(Node(start_connect_x + diff_x/2, start_connect_y + diff_y/2))
                    # new_node_list[-1].parent = index_curr
                    # index_curr += 1
                    # new_node_list.append(Node(start_connect_x + 3*diff_x/4, start_connect_y + 3*diff_y/4))

                    # # TO HERE
                    new_node_list[-1].parent = i
                    break
            for node in new_node_list:
                global count
                nodes_array.insert(count, (float(node.x), float(node.y), float(node.x), float(node.y)), obj=node)
                count += 1
            # print("connection to",  least_smooth[0].y, least_smooth[0].x)
            # print("End of Path")
            return new_node_list

def draw_result(image, result, start, end, node_list, potential_map, show_expansion):
    height = len(image)
    length = len(image[0])

    if show_expansion:
        for x in range(0, length):
            for y in range(0, height):
                if potential_map[y][x] == 1:
                    toPut =  0
                    result.putpixel((x, y), (toPut, toPut, toPut))
                else:
                    toPut = 250 - int(potential_map[y][x]*225)
                    result.putpixel((x, y), (toPut, toPut, toPut))

    for node in node_list:
        round_x = math.floor(node.x)
        round_y = math.floor(node.y)
        for x in range(round_x - 1, round_x + 1):
            for y in range(round_y - 1, round_y + 1):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (255, 0, 0))

    for x in range(start[0] - 3, start[0] + 3):
        for y in range(start[1] - 3, start[1] + 3):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 255, 0))

    for x in range(end[0] - 3, end[0] + 3):
        for y in range(end[1] - 3, end[1] + 3):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 0, 255))


def running_hybridization(image, start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branches_before_each_laplace, visualization):
    global count
    count = 0
    file_name = image
    timer = time.time()
    im = Image.open(image)
    image = np.asarray(im)

    start_comma = start.index(',')
    end_comma = end.index(',')

    start_first_number = int(start[1:start_comma])
    start_second_number = int(start[start_comma+2:len(start)-1])
    end_first_number = int(end[1:end_comma])
    end_second_number = int(end[end_comma+2:len(end)-1])

    start = (start_first_number, start_second_number)
    end = (end_first_number, end_second_number)
    node_list, found_path, potential_map, indices = RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion, branches_before_each_laplace, visualization, output_path)
    end_time = time.time()
    # if visualization:
    #     print("Drawing the result...")
    #     result = im.copy()
    #     height = len(image)
    #     length = len(image[0])
    #     draw_result(image, result, start, end, node_list, potential_map=np.zeros((height, length)), show_expansion=False)
    #     im = Image.open(file_name)
    #     result_images.append(result)
    #     writer = imageio.get_writer(output_path + ".mp4", fps=fps)
    #     for image_filename in result_images:
    #         writer.append_data(np.array(image_filename))
    #     writer.close()

    print(output_path, " took ", str(end_time - timer), " seconds")


images_to_test = ["world3"]
start =  "(1, 1)" #"(254, 180)"#
end =  "(650, 350)" #"(1, 1)"#
iterations = 100000000000000000000
step_size = 1
laplace_iters_to_test = [100]#, 50, 100, 200, 300, 500]
branches_before_each_laplace = [1]#, 5, 10, 25, 50]
tests_of_each = 1
prob_of_choosing_start = 0
show_every_attempted_point = "y"
show_expansion = "y"
fps = 20

for i in range(len(images_to_test)):
    image = images_to_test[i]
    for j in range(len(laplace_iters_to_test)):
        for k in range(len(branches_before_each_laplace)):
            la_place_at_start = laplace_iters_to_test[j]
            la_place_each_time = laplace_iters_to_test[j]
            branch_each_time = branches_before_each_laplace[k]
            for testnum in range(tests_of_each):
                node_list = []
                output_path = str(image) + "_" + str(la_place_each_time) + "laplace_" + str(branch_each_time) + "branches_each_iter_step_size" + str(step_size) +"_testnumber" + str(testnum)
                visualition = False
                running_hybridization(image + ".png", start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branch_each_time, visualition)
                # cProfile.run('running_hybridization(image + ".png", start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branch_each_time, visualition)')
               
                  


log_file.close()
sys.stdout = sys.__stdout__
