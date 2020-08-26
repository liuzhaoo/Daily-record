- 深度学习方面
  
  1. 模型的输入必须是tensor，CHW，输出也是tensor
  2. 数据过少而模型复杂就容易发生过拟合，正则化可以解决。
  3. tensor是内存中连续的内容的集合，不同于python的list：在内存中不连续，由变量名指向它
  4. 直接从数据集加载出来的数据可能是图片类型，需要对其进行转换
  
- 框架
  1. 模型权重的加载或者初始化可以放到模型定义中
  2. 模型的输出一般是类别个分数，代表该类别的置信度。可能还需对其进行计算来完成其他操作（softmax使其分布到0-1之间）。
  3. 在训练之前，往往要对数据进行预处理，比如标准化，裁剪等。这些操作可以在数据集文件里进行，也可以使用torchvision的API
  4. 在Hara等人的实验中，ResNet-18在小数据集（UCF-101,HMBD-53等）上训练发生了过拟合，用相对简单的网络在相对简单的数据集上训练，这个结果是可以预知的。但是如果模型加深一点或者数据集再大一点，甚至两者兼之，效果肯定会变好。
  
- pytorch API

  1. 对输入图像的预处理可以在dataset里留一个transform，然后用torchvision里的transforms：

     ```python
     from torchvision import transforms	
     preprocess = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
     )])
     ```

  2. tensor类型之间的转换

      - 直接指定类型

        ```python
      short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
        double_points = torch.ones(10, 2, dtype=torch.double)
        ```

      - 使用api

        ```python
        double_points = torch.zeros(10, 2).double()  # 直接转换
        short_points = torch.ones(10, 2).short()
        
        double_points = torch.zeros(10, 2).to(torch.double)  # .to
        short_points = torch.ones(10, 2).to(dtype=torch.short)
        ```

  3. 加载预训练模型

      ```python
      model_path = '../data/p1ch2/horse2zebra_0.4.0.pth'
      model_data = torch.load(model_path)
      netG.load_state_dict(model_data)
      ```

  4. 计算数据集图片的均值和标准差，以cifa10为例，加载出来的图片为tensor，在dim3上进行堆叠，然后只留下通道维度，再对各个通道的数据求均值和标准差。

      ​	cifa10 的均值和标准差为：

      > tensor([0.4915, 0.4823, 0.4468])
      >
      > tensor([0.2470, 0.2435, 0.2616])

      ```
      from dataset import cifa10
        abspath = './'
        
        train_dataloader = cifa10(abspath, train=True, transform=transforms.ToTensor())
        imgs = torch.stack([img_t for img_t,t in train_dataloader],dim=3)
        
        imgs.view(3, -1).mean(dim=1)
        imgs.view(3, -1).std(dim=1)
      ```

  5. ```	torch.max``` 需要指定维度，返回两个值，最大值及其index

  6. 假如 $x$ 为tensor ，则```x.mean(dim)``` 就是求对应维度的均值，```x.sum(dim)``` 是求对应维度的数值之和。同时，这个维度会消失。 

  7. transform里的ToTensor除了将pic和narray类型的数据转换为


  ​    

