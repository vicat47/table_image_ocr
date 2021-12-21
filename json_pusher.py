from dataclasses import dataclass, field, asdict
import json, datetime, requests, configparser
from pprint import pprint
from typing import List

from image_read import read_image

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini', encoding="UTF-8")
caffeine_server_url = config.get("caffeine", "CAFFEINE_URL")

@dataclass
class Dish:
    weekday: str = None
    dishes: List[str] = field(default_factory=list)
    cold_dishes: List[str] = field(default_factory=list)
    main_food: List[str] = field(default_factory=list)
    fruit: str = None
    soap: List[str] = field(default_factory=list)
    steamed: List[str] = field(default_factory=list)
    def __post_init__(self, dish_list = []) -> None:
        for index, value in enumerate(dish_list):
            self.set_field(index, value)
    def set_field(self, n: int, value: str):
        if n == 0:
            self.weekday = value
        elif 0 < n <= 4:
            self.dishes.append(value)
        elif 4 < n <= 6:
            self.cold_dishes.append(value)
        elif n == 7:
            self.fruit = value
        elif n == 8:
            self.soap.append(value)
        elif n == 9:
            self.main_food.append(value)
        elif n == 10:
            self.steamed.append(value)
        else:
            print("没有这个菜了，超长")

def load_file(file_path: str) -> list:
    with open(file_path, encoding='UTF-8') as d:
        j = json.loads(d.read())
        pprint(j)
        return j


def save_json(list: list, server_url: str, namespace: str):
    time = datetime.datetime.now()
    weekday = time.weekday()
    current_day = time + datetime.timedelta(-(weekday))
    for d in list:
        id = current_day.strftime("%Y%m%d")
        pprint(id)
        current_day = current_day + datetime.timedelta(1)
        save_caffeine(server_url, namespace, id, d)


def save_caffeine(server_addr:str, namespace:str, id:str, data:dict):
    save_addr = server_addr + namespace + '/' + id
    res = requests.post(url=save_addr, json=data)
    if not res.ok:
        print("保存失败")

if __name__ == '__main__':
    result = read_image('example.jpg')
    dishes = map(lambda row: Dish(row), result)
    save_json(list(map(lambda d: asdict(d), dishes)), caffeine_server_url, 'dishes')