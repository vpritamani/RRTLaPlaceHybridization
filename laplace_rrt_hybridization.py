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

# Must install cupy to run this utilizing GPU
from cupyx.scipy.signal import convolve2d
import cupy as cp

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
def random_point(start, height, length, potential_map, file_name, image, end, prob_of_choosing_start, show_every_attempted_point, show_expansion, start_time, visualization, pointsss, edge_points):
    if(len(edge_points) == 0):
        return (-1, -1, potential_map, edge_points)

    len_per_iter.append(0) 

    time_per_iter.append(time.time() - start_time)

    [new_y, new_x] = edge_points[random.randint(0, len(edge_points)-1)]
    if visualization:
        if(show_every_attempted_point and image[new_y][new_x][0] == 255 and new_y < height - 2 and new_x < length - 2):
            im = Image.open(file_name)
            result = im.copy()
            draw_result(image, result, start, end, node_list, potential_map, show_expansion)
            for x in range(new_x - 3, new_x + 3):
                for y in range(new_y - 3, new_y + 3):
                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                        result.putpixel((x, y), (0, 0, 255))
            result_images.append(result)
    return (new_x, new_y, potential_map, edge_points)

def LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always, node_list_shape):
    # CPU with Numpy array (requires kernel to be numpy as well)
    # potential_map = cv2.filter2D(potential_map, -1, kernel)
    # GPU with Cupy array (requires kernel to be cupy as well)
    potential_map = convolve2d(potential_map, kernel, boundary='symm', mode='same')

    potential_map[end[1]][end[0]] = 0
    potential_map[node_list_shape] = 0
    potential_map[boundary_y, boundary_x] = 1
    potential_map[:, [0, -1]] = 1
    potential_map[[0, -1], :] = 1
    return potential_map

# RRT Algorithm
def RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion, branches_before_each_laplace, visualization, output_path):
    height = len(image)
    length = len(image[0])
    
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

    # Kernel for Laplace equation (numpy)
    # kernel = np.array([[0.00, 0.25, 0.00],
    #                    [0.25, 0.00, 0.25],
    #                    [0.00, 0.25, 0.00]])
    kernel = cp.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])
    
    laplacetime = 0
    randompointtime = 0
    graddescenttime = 0
    
    start_time = time.time()
    height_always = potential_map.shape[0]
    length_always = potential_map.shape[1]

    # For GPU, must convert to cupy
    potential_map = cp.asarray(potential_map)

    for _ in range(la_place_at_start):
        # initial = np.sum(potential_map == 1)
        potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always,node_list_shape)
        # new_len = np.sum(potential_map == 1)
        # print(_, (initial - new_len))
    # For GPU, must convert back to numpy
    potential_map = cp.asnumpy(potential_map)

    laplacetime += time.time() - start_time
    

    while pathFound == False:
        total_iter = total_iter + 1
        if potential_map[start[1]][start[0]] != 1:
            print("found solution")
            time_start = time.time()
            test = try_grad_descent(potential_map, step_size, start[0], start[1], node_list, nodes_array)
            graddescenttime += time.time() - time_start
            if(len(test) == 0):
                sometime = time.time()
                # For GPU, must convert to cupy
                potential_map = cp.asarray(potential_map)
                for _ in range(la_place_each_time):
                    # initial = np.sum(potential_map == 1)
                    potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always, node_list_shape)
                    # new_len = np.sum(potential_map == 1)
                    # print(_, (initial - new_len))
                # For GPU, must convert back to numpy
                potential_map = cp.asnumpy(potential_map)
                laplacetime += time.time() - sometime
                print(la_place_each_time)
                continue
            im = Image.open(file_name)
            result = im.copy()
            node_list.extend(test)
            nearest_nodes = list(nodes_array.intersection((start[0] - 1, start[1] - 1, start[0] + 1, start[1] + 1), objects=True))
            current_node = nearest_nodes[0].object
            found_path = []
            while current_node.parent != -1:
                found_path.append(current_node)
                current_node = node_list[current_node.parent]
            new_smooth = smoothness(found_path)
            print("Smoothness new metric from matlab", new_smooth)
            smoothnessarray.append(new_smooth)

            if(len(found_path) > 1):
                x = np.array([node.x for node in found_path])
                y = np.array([node.y for node in found_path])

                dx = np.gradient(x)
                dy = np.gradient(y)

                ddx = np.gradient(dx)
                ddy = np.gradient(dy)

                squared_second_derivative = ddx**2 + ddy**2

                for i in range(len(found_path)):
                    node = found_path[i]
                    round_x = math.floor(node.x)
                    round_y = math.floor(node.y)
                    for x in range(round_x - 1, round_x + 1):
                        for y in range(round_y - 1, round_y + 1):
                            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                                result.putpixel((x, y), (0, 20 + int(squared_second_derivative[i] * 235 / max(squared_second_derivative)), 0))
            
                
                result.save(output_path + '_found_path.png')
            else:
                print("Proper solution not found")
            laplacetimearray.append(laplacetime)
            randompointtimearray.append(randompointtime)
            graddescenttimearray.append(graddescenttime)
            return node_list


        if(total_iter == iterations):
            print("Iteration limit exceeded.")
            laplacetimearray.append(laplacetime)
            randompointtimearray.append(randompointtime)
            graddescenttimearray.append(graddescenttime)
            return node_list

        newtime = time.time()
        potential_map_as_image = np.uint8(255 * potential_map)  
        potential_map_as_image[(potential_map_as_image == 255) | (potential_map_as_image < 254)] = 0
        potential_map_as_image[potential_map_as_image != 0] = 255
        edge = cv2.Canny(potential_map_as_image, 50, 150)
        edge_points = np.argwhere(edge > 0)
        # print out edge image
        randompointtime += time.time() - newtime
        for am in range(branches_before_each_laplace):
            newtime = time.time()
            new_x, new_y, potential_map, edge_points = random_point(start, height, length, potential_map, file_name, image, end, prob_of_choosing_start, show_every_attempted_point, show_expansion, start_time, visualization, am, edge_points)
            randompointtime += time.time() - newtime
            if (new_x == -1):
                laplacetimearray.append(laplacetime)
                randompointtimearray.append(randompointtime)
                graddescenttimearray.append(graddescenttime)
                return node_list
            
            othertime = time.time()
            new_node_list = try_grad_descent(potential_map, step_size, new_x, new_y, node_list, nodes_array)
            for node in new_node_list:
                y, x = int(node.y), int(node.x)
                if 0 <= y < node_list_shape.shape[0] and 0 <= x < node_list_shape.shape[1]:
                    node_list_shape[y, x] = True
            graddescenttime += time.time() - othertime
            i = i + len(new_node_list)
            node_list.extend(new_node_list)
            if visualization:
                len_per_iter.append(len(new_node_list))
                time_per_iter.append(time.time() - start_time)
                im = Image.open(file_name)
                result = im.copy()
                draw_result(image, result, start, end, node_list, potential_map, show_expansion)
                for x in range(new_x - 3, new_x + 3):
                    for y in range(new_y - 3, new_y + 3):
                        if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                            result.putpixel((x, y), (0, 255, 255))
                result_images.append(result)
        sometime = time.time()
        # For GPU, must convert to cupy
        potential_map = cp.asarray(potential_map)
        for _ in range(la_place_each_time):
            # initial = np.sum(potential_map == 1)
            potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always, node_list_shape)
            # new_len = np.sum(potential_map == 1)
            # print(_, (initial - new_len))
        # For GPU, must convert back to numpy
        potential_map = cp.asnumpy(potential_map)
        laplacetime += time.time() - sometime

def try_grad_descent(potential_map, step_size, new_x, new_y, node_list, nodes_array):


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


    
    poser = new_y
    posec = new_x
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
            a = step_size/maggrad
            poser = poser-a*gradr
            posec = posec-a*gradc
        toAppend = Node(posec, poser)
        new_node_list[-1].parent = index_curr
        index_curr += 1
        new_node_list.append(toAppend)
        limit = limit + 1
        nearest_nodes = None
        if 0 < posec < potential_map.shape[1]-1 and 0 < poser < potential_map.shape[0]-1:
            nearest_nodes = list(nodes_array.intersection((posec - 1 - step_size, poser - 1 - step_size, posec + 1 + step_size, poser + 1 + step_size), objects=True))
        valid = []
        if nearest_nodes:
            for node_found in nearest_nodes:
                valid.append(node_found.object)
        if len(valid) != 0:
            node_to_smoothness = []
            for node_in_valid in valid:
                lst = []
                lst.append(new_node_list[-2])
                lst.append(new_node_list[-1])
                lst.append(node_in_valid)
                if(node_in_valid.parent):
                    lst.append(node_list[node_in_valid.parent])
                node_to_smoothness.append([node_in_valid, smoothness(lst)])
            least_smooth = node_to_smoothness[0]
            for every in node_to_smoothness:
                if least_smooth[1] > every[1]:
                    least_smooth = every
            for i, object in enumerate(node_list):
                if object.equals(least_smooth[0]):
                    new_node_list[-1].parent = i
                    break
            for node in new_node_list:
                global count
                nodes_array.insert(count, (float(node.x), float(node.y), float(node.x), float(node.y)), obj=node)
                count += 1
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
    node_list = RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion, branches_before_each_laplace, visualization, output_path)

    if visualization:
        print("Drawing the result...")
        result = im.copy() 
        height = len(image)
        length = len(image[0])
        draw_result(image, result, start, end, node_list, potential_map=np.zeros((height, length)), show_expansion=False)
        im = Image.open(file_name)
        result_images.append(result)
        writer = imageio.get_writer(output_path + ".mp4", fps=fps)
        for image_filename in result_images:
            writer.append_data(np.array(image_filename))
        writer.close()
        with open(output_path + '.csv', 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(len_per_iter)
            csv_writer.writerow(time_per_iter)
    end_time = time.time()
    timesarray.append(end_time - timer)
    testarray.append(output_path)
    print(output_path)
    print(output_path, " took ", str(end_time - timer), " seconds with ", laplacetimearray[len(laplacetimearray) - 1], " laplace time ",randompointtimearray[len(randompointtimearray) - 1], " random poin with image segtime ", graddescenttimearray[len(graddescenttimearray) - 1], " grad descent time")


def csv_to_graph(file):
    data = pd.read_csv(file + ".csv", header=None)

    values = data.iloc[0].values
    time = data.iloc[1].values

    prefix_sum = values.cumsum()

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(values, marker='o', linestyle=' ')
    plt.title('Original Values')
    plt.xlabel('Iterations')
    plt.ylabel('Length of Branch Added')

    plt.subplot(2, 2, 2)
    plt.plot(prefix_sum, marker='o', linestyle='-')
    plt.title('Cumulative Sum')
    plt.xlabel('Iterations')
    plt.ylabel('Prefix Sum of Length of Branch Added')

    plt.subplot(2, 2, 3)
    plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
    plt.title('Histogram of Branch Lengths')

    plt.xlabel('Branch Length')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.plot(time, prefix_sum, marker='o', linestyle='') 

    plt.title('Time vs Prefix Sum')
    plt.xlabel('Time')
    plt.ylabel('Prefix Sum')

    plt.tight_layout()
    plt.savefig(file + '.png', dpi=300, bbox_inches='tight')  


# TESTS BEING RUN:

images_to_test = ["world3", "world4"]
start = "(1, 1)"
end = "(650, 350)"
iterations = 1000
step_size = 1
laplace_iters_to_test = [50, 100, 200, 300, 500]
branches_before_each_laplace = [1, 5, 10, 25, 50]
tests_of_each = 10
prob_of_choosing_start = 0
show_every_attempted_point = "y"
show_expansion = "y"
fps = 20

avgtestarraybranches = []
avgtestarraylaplaceiters = []
avgtestarraystepsize = []
avgtestarrayimage = []
avgtestarraysmoothness = []
avgtestarraytime = []

for i in range(len(images_to_test)):
    image = images_to_test[i]
    for j in range(len(laplace_iters_to_test)):
        for k in range(len(branches_before_each_laplace)):
            la_place_at_start = laplace_iters_to_test[j]
            la_place_each_time = laplace_iters_to_test[j]
            branch_each_time = branches_before_each_laplace[k]
            for testnum in range(tests_of_each):
                
                len_per_iter = []
                time_per_iter = []
                node_list = []
                result_images = []
                output_path = str(image) + "_" + str(la_place_each_time) + "laplace_" + str(branch_each_time) + "branches_each_iter_step_size" + str(step_size) +"_testnumber" + str(testnum)
                visualition = False
                running_hybridization(image + ".png", start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branch_each_time, visualition)
                if visualition:
                    csv_to_graph(output_path)
                
                testbranchesarray.append(branch_each_time)
                teststepsizearray.append(step_size)
                testlaplaceperarray.append(la_place_each_time)
                testimagearray.append(image)
            avgtestarraybranches.append(branch_each_time)
            avgtestarraylaplaceiters.append(la_place_at_start)
            avgtestarraystepsize.append(step_size)
            avgtestarrayimage.append(image)
            avgtestarraysmoothness.append(sum(smoothnessarray[-tests_of_each:]) / tests_of_each)
            avgtestarraytime.append(sum(timesarray[-tests_of_each:]) / tests_of_each)

step_size = 3

for i in range(len(images_to_test)):
    image = images_to_test[i]
    for j in range(len(laplace_iters_to_test)):
        for k in range(len(branches_before_each_laplace)):
            la_place_at_start = laplace_iters_to_test[j]
            la_place_each_time = laplace_iters_to_test[j]
            branch_each_time = branches_before_each_laplace[k]
            for testnum in range(tests_of_each):
                
                len_per_iter = []
                time_per_iter = []
                node_list = []
                result_images = []
                output_path = str(image) + "_" + str(la_place_each_time) + "laplace_" + str(branch_each_time) + "branches_each_iter_step_size" + str(step_size) +"_testnumber" + str(testnum)
                visualition = False
                running_hybridization(image + ".png", start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branch_each_time, visualition)
                if visualition:
                    csv_to_graph(output_path)
                
                testbranchesarray.append(branch_each_time)
                teststepsizearray.append(step_size)
                testlaplaceperarray.append(la_place_each_time)
                testimagearray.append(image)
            avgtestarraybranches.append(branch_each_time)
            avgtestarraylaplaceiters.append(la_place_at_start)
            avgtestarraystepsize.append(step_size)
            avgtestarrayimage.append(image)
            avgtestarraysmoothness.append(sum(smoothnessarray[-tests_of_each:]) / tests_of_each)
            avgtestarraytime.append(sum(timesarray[-tests_of_each:]) / tests_of_each)


data = [
    ['Test Name'] + testarray,
    ['Image'] + testimagearray,
    ['LaPlace Iters'] + testlaplaceperarray,
    ['Branches'] + testbranchesarray,
    ['Step Size'] + teststepsizearray,
    ['Smoothness'] + smoothnessarray,
    ['Total Time'] + timesarray,
    ['LaPlace Time'] + laplacetimearray,
    ['Random Point from Image Segmentation Time'] + randompointtimearray,
    ['Grad Descent Time'] + graddescenttimearray
]

avgdata = [
    ['Image'] + avgtestarrayimage,
    ['LaPlace Iters'] + avgtestarraylaplaceiters,
    ['Branches'] + avgtestarraybranches,
    ['Step Size'] + avgtestarraystepsize,
    ['Avg Smoothness'] + avgtestarraysmoothness,
    ['Avg Time'] + avgtestarraytime    
]

transposed_data = list(zip(*data)) 
transposed_avgdata = list(zip(*avgdata)) 

with open('october_14th_all_times_compiled.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(transposed_data)
with open('october_14th_avg_times_compiled.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(transposed_avgdata)


log_file.close()
sys.stdout = sys.__stdout__