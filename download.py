# -*- coding: utf-8 -*-
#设置该程序运行时使用的编码格式为utf-8编码
import requests
import os
import re
from threading import Thread
import cv2 as cv
import numpy as np
import random

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
path=r'C:\Users\15848\Desktop\test_2\picture' 
#设置Thread类，准备多线程下载图片
class DownloadTask(Thread):
    #继承
    def __init__(self, keyword, save_dir):
        super().__init__()
        self._keyword = keyword
        self._save_dir = save_dir
    #爬取图片代码
    def run(self):
        n = 1   #图片命名名称
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:66.0) Gecko/20100101 Firefox/66.0',
            'Referer':'https://www.baidu.com',}
        url = 'https://image.baidu.com/search/acjson?'
        # 请求的 url header,存储的是浏览器头部信息
        for pn in range(0,300,30):
            # 请求参数
            param = {'tn': 'resultjson_com',
                    # 'logid': '7603311155072595725',
                    'ipn': 'rj',
                    'ct': 201326592,
                    'is': '',
                    'fp': 'result',
                    'queryWord': self._keyword,
                    'cl': 2,
                    'lm': -1,
                    'ie': 'utf-8',
                    'oe': 'utf-8',
                    'adpicid': '',
                    'st': -1,
                    'z': '',
                    'ic': '',
                    'hd': '',
                    'latest': '',
                    'copyright': '',
                    'word':self._keyword,
                    's': '',
                    'se': '',
                    'tab': '',
                    'width': '',
                    'height': '',
                    'face': 0,
                    'istype': 2,
                    'qc': '',
                    'nc': '1',
                    'fr': '',
                    'expermode': '',
                    'force': '',
                    'cg': '',    # 这个参数没公开，但是不可少
                    'pn': pn,    # 显示：30-60-90
                    'rn': '30',  # 每页显示 30 条
                    'gsm': '1e',
                    '1618827096642': ''
                    }
            # 下载网页源代码
            request = requests.get(url=url, headers=header, params=param)
            request.encoding = 'utf-8'
            html = request.text
            #提取图片url
            image_url_list = re.findall('"thumbURL":"(.*?)",', html, re.S)
            #保存图片
            for image_url in image_url_list:
                image_data = requests.get(url=image_url, headers=header).content
                with open(os.path.join(self._save_dir, f'{self._keyword}_{n}.jpg'), 'wb') as fp:
                    fp.write(image_data)
                n = n + 1
#多线程下载图片
def download():

    print('start_1')
    t1 = DownloadTask('狗','picture')
    t1.start()
    print('start_2')
    t2 = DownloadTask('猫','picture')
    t2.start()

    t1.join()
    t2.join()
        
    print('Get images finished.')
#随机选择图片，并展示
def pick_show():
    train = r'C:\Users\15848\Desktop\test_2\picture' 
    def get_one_image(train): 

        files = os.listdir(train)
        n = len(files)
        ind = np.random.randint(0,n)
        img_dir = os.path.join(train,files[ind])  

        return img_dir  
    
    dir=get_one_image(train)

    #原图
    image = cv.imdecode(np.fromfile(dir, dtype=np.uint8), 1) 
    cv.namedWindow('show', cv.WINDOW_NORMAL)
    cv.imshow("show", image)
    cv.waitKey()
    # 缩放到原来的二分之一，输出尺寸格式为（宽，高）
    x, y = image.shape[0:2]
    img_test1 = cv.resize(image, (int(x / 2), int(y / 2)))
    cv.namedWindow('resize0', cv.WINDOW_NORMAL)
    cv.imshow('resize0', img_test1)
    cv.waitKey()
    #图像垂直翻转
    v_image=cv.flip(image,-1) 
    cv.namedWindow('inv', cv.WINDOW_NORMAL)
    cv.imshow("inv", v_image)
    cv.waitKey()
#改格式
def change():
    for filename in os.listdir(path): #文件夹里不止一张图片，所以要用for循环遍历所有的图片
        if os.path.splitext(filename)[1] != '.jpg': #把path这个路径下所有的文件都读一遍，如果后缀名不是jpg，进行下一步，即imread的读取
            name=os.path.splitext(filename)[1]
            img_path = path + '/' + filename 



            img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), 1) 
            newfilename = filename.replace(name, ".jpg") #用replace函数换成.jpg
            new_path = path + '/' + newfilename 
            print(filename)
            cv.imencode('.jpg', img)[1].tofile(new_path)
            os.remove(img_path)


