- 深度学习方面
  
  1. 模型的输入必须是tensor，CHW，输出也是tensor
  2. 数据过少而模型复杂就容易发生过拟合，正则化可以解决。
  
- 框架
  1. 模型权重的加载或者初始化可以放到模型定义中
  2. 模型的输出一般是类别个分数，代表该类别的置信度。可能还需对其进行计算来完成其他操作（softmax使其分布到0-1之间）。
- pytorch接口
  1. 对输入图像的预处理可以在dataset里留一个transform，然后用torchvision里的transforms：

     ```python
  from torchvision import transforms	
  preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.2
     25]
  )])
     ```

  2. ```torch.max```返回两个值，最大值及其index

  3. 加载预训练模型

     ```python
     model_path = '../data/p1ch2/horse2zebra_0.4.0.pth'
     model_data = torch.load(model_path)
     netG.load_state_dict(model_data)
     ```

  4. 直接从数据集加载出来的数据可能是图片类型，需要对其进行转换