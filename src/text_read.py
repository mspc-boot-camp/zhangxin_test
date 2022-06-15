# coding: utf-8
import os

def createFilelist(images_path, text_save_path):
    # 打开图片列表清单txt文件
    file_name = open(text_save_path, "w")
    # 查看文件夹下的图片
    images_name = os.listdir(images_path)
    # 遍历所有文件
    for eachname in images_name:
        # 按照需要的格式写入目标txt文件
        file_name.write(images_path + '/' + eachname + '\n')
        
    print('生成txt成功！')

    file_name.close()



if __name__ == "__main__":
    # txt文件存放目录
    txt_path = r'C:\Users\15848\Desktop\dog_cat_classification\input' 
    # 图片存放目录
    images_path = r'C:\Users\15848\Desktop\AlexNet\data_name\Cat'
    # 生成图片列表文件命名
    txt_name = 'img_paths.txt'
    if not os.path.exists(txt_name):
        os.mkdir(txt_name)
    # 生成图片列表文件的保存目录
    text_save_path = txt_path + '/' + txt_name
    #生成txt文件
    createFilelist(images_path, text_save_path)
