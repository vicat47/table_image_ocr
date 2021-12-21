import json
import time
import math
import sys
import configparser
from dataclasses import dataclass, field, asdict
from functools import reduce
from typing import List

import cv2
import numpy as np
from aip import AipOcr

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini', encoding="UTF-8")

CELL_MIN_WIDTH = config.getint('image_edge', 'CELL_MIN_WIDTH')
CELL_MIN_HEIGHT = config.getint('image_edge', 'CELL_MIN_HEIGHT')
PIXEL_INCREASE = config.getint('image_edge', 'PIXEL_INCREASE')
COLOR_MINIMUM = config.getint('image_edge', 'COLOR_MINIMUM')

# 百度 ocr api
APP_ID = config.get('baidu_ocr', 'APP_ID')
API_KEY = config.get('baidu_ocr', 'API_KEY')
SECRET_KEY = config.get('baidu_ocr', 'SECRET_KEY')

# 其他配置
WRITE_TO_FOLDER = config.getboolean('image_write', 'WRITE_TO_FOLDER')
BASE_DIR = config.get('image_write', 'BASE_DIR')

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


def read_image(image_path):
    image = cv2.imread(image_path)
    # 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -30)
    write_image('output/cell.jpg', binary)

    rows, cols = binary.shape
    scale = 20
    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    # 腐蚀
    eroded = cv2.erode(binary, kernel, iterations=1)
    # 膨胀
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1) 
    write_image('output/dilated1.jpg', dilatedcol)

    # 识别竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    # 往上延申下
    for i in range(len(dilatedrow) - 1):
        if dilatedrow[i].max() > COLOR_MINIMUM:
            dilatedrow[i - PIXEL_INCREASE: i] = [dilatedrow[i]] * PIXEL_INCREASE
            break
    write_image('output/dilated2.jpg', dilatedrow)

    # 标识交点
    bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
    write_image('output/bitwise.jpg', bitwiseAnd)

    # 标识表格
    merge = cv2.add(dilatedcol, dilatedrow)
    write_image('output/add.jpg', merge)
    # 通过 Mask 去除一些东西
    bitwise_sub = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(cv2.dilate(merge, np.ones((4, 4), np.uint8), iterations=1)))
    write_image('output/sub.jpg', bitwise_sub)

    # 识别黑白图中的白色点
    ys, xs = np.where(bitwiseAnd > 0)
    mylisty = []
    mylistx = []

    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值，我只取最后一点
    i = 0
    myxs = np.sort(xs)
    print('myxs', myxs)
    for i in range(len(myxs) - 1):
        if myxs[i + 1] - myxs[i] > CELL_MIN_WIDTH:
            mylistx.append(myxs[i])
        i = i + 1
    mylistx.append(myxs[i])
    print('纵向：', mylistx)
    print(len(mylistx))

    i = 0
    myys = np.sort(ys)
    print('myys', myys)

    for i in range(len(myys) - 1):
        if myys[i + 1] - myys[i] > CELL_MIN_HEIGHT:
            mylisty.append(myys[i])
        i = i + 1
    mylisty.append(myys[i])
    print('横向', mylisty)
    print(len(mylisty))

    index = 0
    rows = []
    for i in range(index, len(mylisty) - 1):
        # current_dish = Dish()
        current_row = []
        for n in range(len(mylistx) - 1):
            res = do_cell(n, i, mylistx, mylisty, bitwise_sub)
            if len(res.get('words_result')) < 1:
                current_row.append('')
                continue
            current_row.append(reduce(lambda x, y: x + y, map(lambda x: x.get("words"), res.get('words_result')), ""))
            # current_dish.set_field(n, reduce(lambda x, y: x + y, map(lambda x: x.get("words"), res.get('words_result')), ""))
        # dish_list.append(current_dish)
        rows.append(current_row)
    print(rows)
    return rows


def do_cell(x: int, y: int, x_cross_list: list, y_cross_list: list, img):
    """
    对每个单元格进行处理......
    """
    # 缩小ROI范围，自定义浮动，让边缘切的更好一点
    roi = img[y_cross_list[y]:y_cross_list[y + 1] - 3, x_cross_list[x]:x_cross_list[x + 1] - 3]
    img_name = str(y) + '_' + str(x) + '.jpg'
    img_path = 'output/' + img_name
    concated_row = split_projection_list(roi, h_project(roi))
    write_image(img_path, concated_row)
    # 调用 百度 OCR 接口，百度有限制 QPS = 1
    # res = get_baidu_ocr_result(img_path)
    res = get_baidu_ocr_result(np.array(cv2.imencode('.png', concated_row)[1]).tobytes())
    return res


def h_project(img):
    '''
    这是水平方向投影的方法，返回投影后的对象
    '''
    h, w = img.shape
    hprojection = np.zeros(img.shape, dtype=np.uint8)
    h_h = [0]*h
    for j in range(h):
        for i in range(w):
            if img[j,i] > 200:
                h_h[j] += 1
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 0
    return h_h


def split_projection_list(img, project_list, min_value=0) -> np.ndarray:
    start = 0
    end = None
    '规定行高，方便后面进行验证'       
    max_line_height = 0
    split_list = []
    for idx, value in enumerate(project_list):
        if value > min_value:
            end = idx
        else:
            if end is not None and end - start < 1:
                end = None
            if end is not None:
                split_list.append((start, end))
                max_line_height = max(max_line_height, end - start)
                end = None
            start = idx

    index = 0
    while index < len(split_list):
        s, e = split_list[index]
        curr_line_height = e - s
        # show_image_in_window("拼接后的图片", img[s:e, ])
        if (curr_line_height < (max_line_height / 5) * 4):
            if (len(split_list) - 1 > index):
                '有下一行'
                if (split_list[index + 1][1] - split_list[index + 1][0] < (max_line_height / 5) * 4):
                    split_list[index] = (split_list[index][0], split_list[index + 1][1])
                    del split_list[index + 1]
                    # show_image_in_window("拼接后的图片", img[split_list[index][0]:split_list[index][1], ])
                    continue
            endpoint = max_line_height / 2 - (e - s) / 2
            updated_line = (math.floor(s - endpoint), math.floor(e + endpoint))
            last_row_end = 0 if index <= 0 else split_list[index - 1][1]
            next_row_start = updated_line[1] + 1 if index >= len(split_list) - 1 else split_list[index + 1][0]
            if (updated_line[0] < last_row_end or updated_line[1] > next_row_start):
                index = index + 1
                continue
            split_list[index] = updated_line
        index = index + 1

    split_rows = []
    for row in split_list:
        splited_img = img[row[0]:row[1], ]
        split_rows.append(splited_img)
        # show_image_in_window("切割后的投影", splited_img)
    '''
    原生的 hconcat 只能拼接高度一样的图
    '''
    img_result = my_hconcat(split_rows)
    # show_image_in_window("拼接后的图片", img_result)
    return img_result


def my_hconcat(imgs:list):
    height = max(map(lambda x: x.shape[0], imgs))
    width = sum(map(lambda x: x.shape[1], imgs))
    res_image = np.zeros((height, width), np.uint8)
    start_x_index = 0
    end_x_index = 0
    for row in range(0, len(imgs)):
        end_x_index = start_x_index + imgs[row].shape[1]
        res_image[0:imgs[row].shape[0], start_x_index:end_x_index] = imgs[row]
        start_x_index = end_x_index
    return res_image

def get_baidu_ocr_result(image_content):
    """ 百度 OCR 接口，百度有限制 QPS = 1 """
    # cell_img = get_file_content(image_path)
    res = client.basicGeneral(image_content)
    # TODO: 处理异常
    print(res)
    time.sleep(1)
    return res
    # return {'words_result': [{'words': 'test'}, {'words': 'data'}], 'words_result_num': 2, 'log_id': 12345678900000}


def get_file_content(file_path):
    """ 读取图片 """
    with open(file_path, 'rb') as fp:
        return fp.read()


def write_image(image_path, image):
    '''
    向文件中写入图片，根据 WRITE_TO_FOLDER 开关变量进行处理
    '''
    base_dir = BASE_DIR if BASE_DIR != '' else './'
    if WRITE_TO_FOLDER:
        cv2.imwrite(base_dir + image_path, image)

def show_image_in_window(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = sys.argv
    d = read_image('example.jpg')
    with open("output/dishes.json", "w", encoding="utf-8") as j:
        json.dump(list(map(lambda x: asdict(x), d)), j, ensure_ascii=False)
