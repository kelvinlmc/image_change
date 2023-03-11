'''
import torch
classes = [
    "0",
    "8",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "5",
]
a=torch.tensor([ 0.0538,  0.1298,  0.0438, -0.0213, -0.0079, -0.1477, -0.0053,  0.0796,-0.0144,  0.0296])
b=torch.tensor(5)
print(b)
print(b.item())
'''
'''
#print(torch.argmax(a))
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from PIL import Image
import time
toPIL = transforms.ToPILImage()
train_dataset=datasets.MNIST(root='./data',transform=transforms.ToTensor())
train_dataloader=DataLoader(dataset=train_dataset,batch_size=1,shuffle=True, num_workers=0)
for batch,(X,y) in enumerate(train_dataloader):
    print(X.size())
    X1=torch.squeeze(X)
    print(X1.size())
    image = toPIL(X1)
    image.show()
    time.sleep(3)
'''

'''
#读取图片并且显示像素,显示不同通道的图片
from PIL import Image
from  torchvision import transforms
data_transfom=transforms.Compose([
    transforms.ToTensor()
])
image=Image.open('./data/dog.jpg')
data_dog=data_transfom(image)
print(data_dog.size())
print(data_dog)
#R通道
red_dog=data_dog[0]
toPIL=transforms.ToPILImage()
imag= toPIL(red_dog)
#imag.show()
#G通道
green_dog=data_dog[2]
toPIL=transforms.ToPILImage()
imag1= toPIL(green_dog)
imag1.show()
'''
'''
#计算损失正确理解
import torch
from torch import tensor
from torch import nn
a=tensor([[ 0.0815, -0.0953, -0.0539, -0.1506, -0.1855,  0.1024, -0.0572,  0.1364,
          0.2534, -0.1784],
        [ 0.0803, -0.0952, -0.0549, -0.1512, -0.1856,  0.1023, -0.0582,  0.1370,
          0.2538, -0.1791],
        [ 0.0815, -0.0956, -0.0548, -0.1502, -0.1858,  0.1028, -0.0570,  0.1354,
          0.2537, -0.1777],
        [ 0.0810, -0.0953, -0.0536, -0.1499, -0.1857,  0.1016, -0.0565,  0.1362,
          0.2537, -0.1775],
        [ 0.0808, -0.0958, -0.0554, -0.1506, -0.1859,  0.1033, -0.0561,  0.1350,
          0.2523, -0.1780],
        [ 0.0806, -0.0953, -0.0546, -0.1505, -0.1863,  0.1032, -0.0569,  0.1352,
          0.2531, -0.1766],
        [ 0.0794, -0.0947, -0.0547, -0.1515, -0.1856,  0.1021, -0.0565,  0.1370,
          0.2527, -0.1778],
        [ 0.0801, -0.0955, -0.0552, -0.1496, -0.1869,  0.1041, -0.0567,  0.1355,
          0.2534, -0.1763],
        [ 0.0803, -0.0951, -0.0546, -0.1506, -0.1862,  0.1037, -0.0579,  0.1360,
          0.2545, -0.1772],
        [ 0.0820, -0.0964, -0.0538, -0.1501, -0.1855,  0.1018, -0.0568,  0.1366,
          0.2528, -0.1785],
        [ 0.0791, -0.0943, -0.0551, -0.1519, -0.1859,  0.1029, -0.0572,  0.1353,
          0.2540, -0.1768],
        [ 0.0798, -0.0949, -0.0554, -0.1508, -0.1860,  0.1033, -0.0577,  0.1354,
          0.2548, -0.1758],
        [ 0.0809, -0.0960, -0.0538, -0.1507, -0.1856,  0.1032, -0.0571,  0.1361,
          0.2534, -0.1770],
        [ 0.0794, -0.0950, -0.0545, -0.1501, -0.1870,  0.1041, -0.0583,  0.1370,
          0.2546, -0.1790],
        [ 0.0807, -0.0957, -0.0565, -0.1499, -0.1864,  0.1035, -0.0562,  0.1340,
          0.2539, -0.1768],
        [ 0.0808, -0.0954, -0.0533, -0.1485, -0.1866,  0.1033, -0.0588,  0.1373,
          0.2543, -0.1784]])
b=tensor([6, 4, 2, 1, 7, 7, 9, 2, 0, 7, 4, 0, 4, 9, 4, 6])
c=tensor([[ 0.0815, -0.0953, -0.0539, -0.1506, -0.1855,  0.1024, -0.0572,  0.1364,
          0.2534, -0.1784]])
d=tensor([6])
loss_fun=nn.CrossEntropyLoss()
print(loss_fun(c,d))
'''


'''nn.CrossEntropyLoss()要求第一个参数是二维张量第一维是批量个数，第二维是批量里面判断每一个目标分别对应的概率值，第二个参数为类 
批量里面有多少，那么第二个参数就有多少类别，因为前后两个参数是一一对应的，第一个参数里的第一个目标的类别的概率对应第二个参数的第一个数，
而这个数就是这个物体的真实类别''
print(loss_fun(c,d))
#print(data_dog.size())
#image=Image.open('./data/dog.jpg')
#image.show()
#plt.imshow(img,cmap='gray')

from torch import nn
'''
'''
#计算损失(错误理解)
import torch
loss_fun=nn.CrossEntropyLoss()
a=torch.tensor([7])
b=torch.tensor([3])
cur_loss = loss_fun(a,b)
print("损失值为",cur_loss)'''

'''
#扩展维度
from PIL import Image
from  torchvision import transforms
import torch
data_transfom=transforms.Compose([
    transforms.ToTensor()
])
image=Image.open('./data/dog.jpg')
data_dog=data_transfom(image)
print(data_dog.size())
dog=torch.unsqueeze(data_dog,0)
print(dog.size())
'''


#修改图片像素大小
from PIL import Image
from torchvision import transforms
data_transform=transforms.Compose([
    transforms.ToTensor()])
image=Image.open('./data/dog.jpg')
image1=image.resize((224,224),1)  #resize()第一个参数是想要的长宽，第二个参数是改变方式有好几种，缩小则直接丢弃像素放大则少的像素全用0补齐
image_size=data_transform(image)  #reshape()则是可以修改通道的，反正前后像素相等就行否则会报错
image1_size=data_transform(image1)
print(image_size.size())
print(image1_size.size())
image1.show()

'''
#使用make_grid函数
def show_make(tensor):
    image=make_grid(tensor,nrow=5,padding=2)#make_grid()可以将思维张量排列展示，前面的batch_size自动捕捉后面为图片参数
    #image.permute(1, 2, 0).squeeze()
    plt.imshow(image.permute(1, 2, 0)) #imshow()是将tensor格式转换成图片,里面的参数必须是三维也就是通道和长宽，而且长宽在前面通道在后面
    plt.show()
s=torch.randn(6,3,224,224)
print(s.size())
#show_make(s)
plt.imshow(torch.randn(3,28,28).permute(1, 2, 0))
plt.show()   #括号里没有对象直接是plt.show()

'''

'''
1.要将图像用Image库打开后再去读取其像素就可以打印出像素值，transforms.ToTensor()的操作对象有PIL格式的图像以及numpy
2.torchvision.transforms.ToTensor或者ToPIL原始图片转化为tensor是归一化处理后的所以要还原其像素值后才可以显示图片
3.全连接层的分类原理：其实全连接层就是一个线性变换，如果说送入模型的批次为1那么输出的全连接层的神经元个数就是判断这张图片
类别的参数w，当然理论上全连接层数越多那么效果越好，因为这样的话就是在找决定类别的参数，如果可以训练很多次那么就可以找到一组参数使得能
很好的符合这个判别标准，就拿最后两层全连接来说，比如Lenet最后一层10个类别也就是10个神经元，如果批次为1，也就是送进模型的只有一张
图片那么经过特征提取后只有前面84个数字（1x84）然后经过10个参数线性处理得到一个分值然后在进行概率化就可以得到这个图片的概率
'''
