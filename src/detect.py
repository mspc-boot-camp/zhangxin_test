<<<<<<< HEAD
import torch
import csv
import time
from build_network import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



ROOT_TEST = r'C:\Users\15848\Desktop\classification\dog_cat_classification\img'



# 将图像的像素值归一化到【-1， 1】之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
    ])


val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)


val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)

# 加载模型
model.load_state_dict(torch.load(r"C:\Users\15848\Desktop\classification\dog_cat_classification\weights\best_model.pth"))

# 获取预测结果
classes = [
    "0",  #cat
    "1",  #dog
]

trip_list = []
# 进入到验证阶段
model.eval()
for i in range(10):  #测试图片数量,需要根据实际情况修改
    T1=time.time()
    x, y = val_dataset[i][0], val_dataset[i][1]
 
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():     
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')
        T2=time.time()
        run_time=T2-T1
        #将数据保存在列表中，miles_driven,gallons_used,mpg三个数据时手动输入赋值的
        trip_list=(predicted,str(run_time))
 		#以写的模式打开文件
        with open(r"C:\Users\15848\Desktop\classification\dog_cat_classification\output\out_results.csv",'a',newline='') as csvfile:
            writer = csv.writer(csvfile)

            #将数据写入csv文件
            writer.writerow(trip_list)
=======
import torch
from build_network import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

ROOT_TRAIN = r'C:/Users/15848/Desktop/AlexNet/data/train'
ROOT_TEST = r'C:/Users/15848/Desktop/AlexNet/data/val'



# 将图像的像素值归一化到【-1， 1】之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)

# 加载模型
model.load_state_dict(torch.load(r"C:\Users\15848\Desktop\dog_cat_classification\weights\best_model.pth"))

# 获取预测结果
classes = [
    "cat",
    "dog",
]

# 把张量转化为照片格式
show = ToPILImage()

# 进入到验证阶段
model.eval()
for i in range(10):
    x, y = val_dataset[i][0], val_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')
>>>>>>> ff1a5f47fc24f8533962a06ba9065315c3522098
