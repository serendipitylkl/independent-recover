import h5py   #这是一个库，跟数据有关的库
import glob   #glob是python自带的一个操作文件的相关模块，用它可以查找符合特定规则的文件路径名
import numpy as np #NumPy（Numerical Python）是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵
import torch.utils.data as data #这个package包含数据读取预处理的一些类

class pix2pix(data.Dataset):
  def __init__(self, root):
    self.root = root

  def __getitem__(self, index):
    file_name=self.root+'/'+str(index)+'.h5'
    f=h5py.File(file_name,'r')
    haze_image=f['haze'][:]
    trans_map=f['trans'][:]
    ato_map=f['ato'][:]
    GT=f['gt'][:]

    haze_image=np.swapaxes(haze_image,0,2)#np.swapaxes 交换数组的两个轴
    trans_map=np.swapaxes(trans_map,0,2)
    ato_map=np.swapaxes(ato_map,0,2)
    GT=np.swapaxes(GT,0,2)

    haze_image=np.swapaxes(haze_image,1,2)
    trans_map=np.swapaxes(trans_map,1,2)
    ato_map=np.swapaxes(ato_map,1,2)
    GT=np.swapaxes(GT,1,2)

    return haze_image, GT,  trans_map, ato_map

  def __len__(self):
    train_list=glob.glob(self.root+'/*h5')#返回所有匹配的文件路径列表。它只有一个参数pathname，
    # 定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。下面是使用glob.glob的例子：
    return len(train_list)

