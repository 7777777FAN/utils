


import cv2 as cv
from PIL import Image, ImageDraw, ImageOps
import numpy as np


# 影像裁剪示意图
sat_path = r"./in_images/sat.png"
# 掩膜裁剪示意图
graph_vis_path = r"./in_images/graph.png"
# 图裁剪示意图
mask_path = r"./in_images/mask.png"

img_size = 1024
crop_size = 256
overlap = 0.1
scale = 4
interval = 20   # pixels

def image_pad(sat):
    step = int(crop_size * (1 - overlap))
    counter = img_size / step
    if counter - int(counter) <= 1e-5:
        counter = int(counter)
    else:
        counter = int(counter) + 1
    target_height =  (counter - 1) * step + crop_size
    border = int((target_height - img_size) / 2)
    sat = Image.fromarray(sat)
    # pad_sat = ImageOps.pad(sat, (target_width, target_height))  # 维持宽高比(resize)的情况下视情况进行填充
    pad_sat = ImageOps.expand(sat, border)

    return  pad_sat, counter


def overlap_crop(mode="no_pad"):
    sat = np.array(cv.imread(sat_path).copy())
    graph = np.array(cv.imread(graph_vis_path).copy())
    mask = np.array(cv.imread(mask_path).copy())
    
    colors = [(0, 0, 255), (255, 0, 0)]
    upper_bound = img_size
    step = int(crop_size * (1 - overlap))

    def draw_and_crop():
        nonlocal sat, graph, mask
        if "pad" == mode:
            sat, counter = image_pad(sat)
            sat = np.array(sat)
            graph = np.array(image_pad(graph)[0])
            mask = np.array(image_pad(mask)[0])
        else:
            counter = upper_bound / step
            if counter - int(counter) <= 0.25:  # 有可能恰好被整除或者剩得太少
                counter = int(counter)
            else:
                counter = int(counter) + 1   # 多出来一点，有这么多个slice
        seamless_H = counter * crop_size
        H_with_interval = seamless_H + interval * (counter - 1)
        
        sat_draw = np.array(sat.copy())
        
        new_sat   = np.full((H_with_interval, H_with_interval, 3), fill_value=[255, 0, 0], dtype=np.uint8)
        new_graph = np.full((H_with_interval, H_with_interval, 3), fill_value=[255, 0, 0], dtype=np.uint8)
        new_mask  = np.full((H_with_interval, H_with_interval, 3), fill_value=[255, 0, 0], dtype=np.uint8)
        
        max_x, max_y = sat_draw.shape[1], sat_draw.shape[0]
        x_counter = 0
        for x in range(0, upper_bound, step):   # 列
            color = colors[0]
            new_x = x_counter * (crop_size + interval)
            x_counter += 1
            
            if x + crop_size > max_x:    # 倒多不少嘞剩些，重叠率只能大点儿咯，pad模式不会出现这种情况
                x = x - (x + crop_size - max_x) 
            col_start, col_end = (x, 0), (x, max_y)
            col_crop_start, col_crop_end = (x+crop_size, 0), (x+crop_size, max_y)
            cv.line(sat_draw, col_start, col_end, color=color, thickness=6, lineType=cv.LINE_8)
            cv.line(sat_draw, col_crop_start, col_crop_end, color=color, thickness=6, lineType=cv.LINE_8)
            
            y_finished = False
            y_counter = 0
            for y in range(0, upper_bound, step):
                color = colors[0]
                new_y = y_counter * (crop_size + interval)
                y_counter += 1
                
                if y + crop_size > max_y:    # pad模式不会出现这种情况
                    y = y - (y + crop_size - max_y)
                row_start, row_end = (0, y), (max_x, y)
                row_crop_start, row_crop_end = (0, y + crop_size), (max_x, y+crop_size)
                
                if not y_finished:
                    cv.line(sat_draw, row_start, row_end, color=color, thickness=6, lineType=cv.LINE_8)
                    cv.line(sat_draw, row_crop_start, row_crop_end, color=color, thickness=6, lineType=cv.LINE_8)
                    
                new_sat[new_y:new_y+crop_size, new_x:new_x+crop_size, :] = sat[y:y+crop_size, x:x+crop_size, :]
                new_graph[new_y:new_y+crop_size, new_x:new_x+crop_size, :] = graph[y:y+crop_size, x:x+crop_size, :]
                new_mask[new_y:new_y+crop_size, new_x:new_x+crop_size, :] = mask[y:y+crop_size, x:x+crop_size, :]
            y_finished = True

        cv.imwrite("./out_images/overlap_crop_sat.png", new_sat)
        cv.imwrite("./out_images/overlap_crop_graph.png", new_graph)
        cv.imwrite("./out_images/overlap_crop_mask.png", new_mask)
    
        return sat_draw

    sat_schematic = draw_and_crop()
          
    cv.imwrite("./out_images/overlap_schematic_sat.png", sat_schematic)


def seamless_crop():
    sat = np.array(cv.imread(sat_path).copy())
    graph = np.array(cv.imread(graph_vis_path).copy())
    mask = np.array(cv.imread(mask_path).copy())
    
    H, W = sat.shape[0], sat.shape[1]
    increment = interval * int(img_size / crop_size)
    new_sat = np.full((H+increment, W+increment, 3), fill_value=[0, 0, 0], dtype=np.uint8)
    new_graph = np.full((H+increment, W+increment, 3), fill_value=[0, 0, 0], dtype=np.uint8)
    new_mask = np.full((H+increment, W+increment, 3), fill_value=[255, 0, 0], dtype=np.uint8)
    sat_draw = sat.copy()
    
    row_counter = 0
    col_finished = False
    for row in range(0, img_size, crop_size):
        if row > 0:
            row_counter += 1
            new_row = row + interval * row_counter
        else:
            new_row = row
        cv.line(sat_draw, (0, row), (img_size, row), color=[0, 0, 255], thickness=8)
        
        col_counter = 0

        for col in range(0, img_size, crop_size):
            if col > 0:
                col_counter += 1
                new_col = col + interval * col_counter
            else:
                new_col = col
            # print(row, col)
            if not col_finished:
                cv.line(sat_draw, (col, 0), (col, img_size), color=[0, 0, 255], thickness=8)
            
            new_sat[new_row:new_row+crop_size, new_col:new_col+crop_size, :] = sat[row:row+crop_size, col:col+crop_size, :]
            new_graph[new_row:new_row+crop_size, new_col:new_col+crop_size, :] = graph[row:row+crop_size, col:col+crop_size, :]
            new_mask[new_row:new_row+crop_size, new_col:new_col+crop_size, :] = mask[row:row+crop_size, col:col+crop_size, :]
        col_finished = True

    # 最右边和最下边的两条线
    cv.line(sat_draw, (img_size, 0), (img_size, img_size), color=[0, 0, 255], thickness=8)
    cv.line(sat_draw, (0, img_size), (img_size, img_size), color=[0, 0, 255], thickness=8)
    
    cv.imwrite("./out_images/seamless_schematic_sat.png", sat_draw)

    cv.imwrite("./out_images/seamless_crop_sat.png", new_sat)
    cv.imwrite("./out_images/seamless_crop_graph.png", new_graph)
    cv.imwrite("./out_images/seamless_crop_mask.png", new_mask)



if __name__ == "__main__":
    seamless_crop()
    overlap_crop("no_pad")
    # overlap_crop("pad")
    


