import cv2
import shutil
#文件test.txt
def read_num_txt():
	files=open("input\img_paths.txt","r+")

	#先读取一行
	str=files.readline()
	num=1

	#循环读取
	while str:
		path = str.strip("\n")
		#print(str)
		shutil.copy(path, r'C:\Users\15848\Desktop\classification\dog_cat_classification\img')
		str =files.readline()
		num=num+1
	files.close()
	return num
