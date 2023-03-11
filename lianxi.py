import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.manual_seed(0)

def sive_tensor_images(image_tensor,name, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(fname=name, figsize=[10, 10])
    #plt.show()
'''
#连续保存图片
for i in range(3):
    a=torch.ones(6,1,2,2)
    #name="./data/pig{}.png".format(i)
    sive_tensor_images(a,"./data/pig{}.png".format(i),6,(1,2,2))
    #size = (1, 28, 28)
    #a = torch.tensor([128, 784])
    #print(a.view(-1, *size))
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
