# -*- coding: utf-8 -*-
import requests
import os
import re
from threading import Thread
import cv2 as cv
import numpy as np
import random

#print Chinese prepare
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

#picture save file path
path=r'C:\Users\15848\Desktop\test_2\picture' 

#set Thread class，read to download picture with multithreading
class DownloadTask(Thread):
    #inherit thread class
    def __init__(self, keyword, save_dir):
        super().__init__()
        self._keyword = keyword
        self._save_dir = save_dir
    #python crawler
    def run(self):
        n = 1   #the number to name the pictures

        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:66.0) Gecko/20100101 Firefox/66.0',
            'Referer':'https://www.baidu.com',}
        url = 'https://image.baidu.com/search/acjson?'
        #  url header
        for pn in range(0,300,30):
            #parameters
            param = {'tn': 'resultjson_com',
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
                    'cg': '',    
                    'pn': pn,    # sum：30-60-90
                    'rn': '30',  # each page 30
                    'gsm': '1e',
                    '1618827096642': ''
                    }
            # download th source code
            request = requests.get(url=url, headers=header, params=param)
            request.encoding = 'utf-8'
            html = request.text
            #extract the url
            image_url_list = re.findall('"thumbURL":"(.*?)",', html, re.S)
            #save pictures
            for image_url in image_url_list:
                image_data = requests.get(url=image_url, headers=header).content
                with open(os.path.join(self._save_dir, f'{self._keyword}_{n}.jpg'), 'wb') as fp:
                    fp.write(image_data)
                n = n + 1

#download picture with multithreading
def download():

    print('开始狗狗图片爬虫')
    sys.stdout.flush()
    p1 = DownloadTask('狗','picture')
    p1.start()
    print('开始猫猫图片爬虫')
    p2 = DownloadTask('猫','picture')
    sys.stdout.flush()
    p2.start()

    p1.join()
    p2.join()
        
    print('图片爬虫完成')

#random selecte pictures,process,display
def pick_show():

    print('随机选择图片显示开始')
    
    def get_one_image(path): 

        files = os.listdir(path)
        n = len(files)
        ind = np.random.randint(0,n)
        img_dir = os.path.join(path,files[ind])  

        print('随机选择的是图片',os.path.basename(img_dir),flush=True) #show which one is selected
        return img_dir  
    
    dir=get_one_image(path)
    
    #原图
    image = cv.imdecode(np.fromfile(dir, dtype=np.uint8), 1) 
    cv.imshow("original", image)
    cv.waitKey()
    # 缩放到原来的二分之一，输出尺寸格式为（宽，高）
    x, y = image.shape[0:2]
    img_test1 = cv.resize(image, (int(x / 2), int(y / 2)))
    cv.imshow('zoom', img_test1)
    cv.waitKey()
    #图像垂直翻转
    v_image=cv.flip(image,-1) 
    cv.imshow("flip vertical", v_image)
    cv.waitKey()
    print('图片显示结束')

#change the format
def change():
    print('图片格式修改开始')
    number=0 #record number
    for filename in os.listdir(path): #test all pictures
        if os.path.splitext(filename)[1] != '.jpg': #select
            name=os.path.splitext(filename)[1]
            print('修改的是',filename)    #show which one gonne be change

            img_path = path + '/' + filename 
            img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), 1) 

            newfilename = filename.replace(name, ".jpg") #replace with .jpg
            new_path = path + '/' + newfilename 
            cv.imencode('.jpg', img)[1].tofile(new_path)

            os.remove(img_path)   #delete .png
            number=number+1
    print('图片格式修改完成,一共修改了',number,'个')
if __name__ == "__main__":
    download()
    pick_show()
    change()