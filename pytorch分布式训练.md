# 前言

本文对使用pytorch进行分布式训练（单机多卡）的过程进行了详细的介绍，附加实际代码，希望可以给正在看的你提供帮助。本文分三个部分展开，分别是:

1. 先验知识
2. 使用过程框架
3. 代码解析

若想学习分布式的部署，看完本文就足够了，但为了读者能了解更多细节，我在第一部分的每个模块都加了对应的官方文档的链接。

同时，我正在进行PyTorch官方文档的翻译工作，除了对其进行便于理解的翻译，还添加了我的解释。项目地址：https://github.com/liuzhaoo/Pytorch-API-and-Tutorials-CN，欢迎各位下载使用！

## 一、先验知识

分布式训练涉及到pytorch的很多API，这里对它们进行简单的介绍，其中重点为第三节DataLoader。若想直接看到使用方法，请看第二部分。

### 1.DataParallel 和DistributedDataParallel（DDP）

此两种方法都可以实现多GPU并行训练，但是后者更快，同时需要写更多代码，而`DataParallel`只需一行代码就可以搞定。尽管如此，还是建议使用`DistributedDataParallel`，建议参考[官方介绍](https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead)。

如下，只需在将model加载到device（`model.to(device)`）之后，加上以下代码即可

- `net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])`

#### 本文极力推荐DDP方法，下文也都是对DDP的说明：

DDP为基于`torch.distributed`的分布式数据并行结构，工作机制为：在batch维度上对数据进行分组，将输入的数据分配到指定的设备（GPU）上，从而将程序的模型并行化。对应的，每个GPU上会复制一个模型的副本，负责处理分配到的数据，在后向传播过程中再对每个设备上的梯度进行平均。

在这里贴上官方文档，供读者进行更详细的了解：[DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)

**以下是使用方法：**

在每个有N个GPU 的主机上，都应该创建N个进程。同时确保每个进程分别在从0到N-1的单独的GPU上工作。因此，应该分别指定工作的GPU：

```python
>>> torch.cuda.set_device(i) # i为0 - N-1
```

在每个进程中，参考以下内容来构建模块

```python
>>> from torch.nn.parallel import DistributedDataParallel
>>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
>>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)
```

为了在每个节点上产生多个进程，可以使用`torch.distributed.launch`或[`torch.multiprocessing.spawn`](https://pytorch.org/docs/stable/multiprocessing.html)

### 2. torch.distributed

`torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='')`

> `torch.distributed`包为在一台或多台机器上运行的多个计算节点上的多进程并行结构提供PyTorch支持和通信原语。 [`torch.nn.parallel.DistributedDataParallel()`](https://pytorch.apachecn.org/docs/1.2/nn.html#torch.nn.parallel.DistributedDataParallel)类就是基于此功能构建的，作为任何PyTorch模型的包装来提供同步分布式训练。这不同于 [`Multiprocessing package - torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html) 和 [`torch.nn.DataParallel()`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) 提供的并行结构，因为它支持多台联网的机器而且用户必须显式地为每个进程启动主要训练脚本的副本。

以上叙述来自pytorch[官方文档](https://pytorch.org/docs/stable/distributed.html)，点击链接可以查看详细内容。此教程中只涉及到此包的初始化，因此不对其他内容再做介绍。

**`torch.distributed`初始化**

目前支持三种初始化方式：TCP初始化，共享文件初始化以及环境变量初始化。

一般使用TCP初始化，使用GPU时backend一般设置为'nccl'：

```python
import torch.distributed as dist
# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)
```

### 3. DataLoader

[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)类是PyTorch数据加载功能的核心，此类中的很多参数都是数据并行时所需要的，本节将对它进行详细的介绍。

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```

- **dataset**，即获取的原始数据集，pytorch支持两种不同类型的数据集

  1. `map-style datasets`：一种映射型的数据集，使用`__getitem__()` 和 `__len__()`协议，表示一种从indices/keys（可能为非整型）到数据样本的映射

     比如有这样一个数据集，当访问 `dataset[idx]`时，可以从磁盘上的文件夹读取到第`idx`个图像以及与它相关的标签。

  2. `iterable-style datasets`这类数据集是 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 的子类的一个实例，使用 `__iter__()`协议，表示可在数据样本上迭代。这种类型的数据集特别适合于很难甚至无法进行随机读取，以及BatchSize的大小取决于获取的数据的情况。

     比如调用 `iter(dataset)`时，可以返回从数据库、远程服务器读取的数据流，甚至实时生成的日志。

  我们使用的大部分数据集都是`map-style`类型的数据集

- **sampler，batch_sampler及shuffle**

  这里主要为关于`map-style`的介绍。

  介绍这几个个参数之前，需要认识另一种类

  > <font color='red'>CLASS</font>  `torch.utils.data.Sampler`(*data_source*)

  同种类型的类还有`torch.utils.data.SequentialSampler`，`torch.utils.data.RandomSampler`， `torch.utils.data.SubsetRandomSampler` `torch.utils.data.WeightedRandomSampler` `torch.utils.data.BatchSampler`，`torch.utils.data.distributed.DistributedSampler`。

  这些类的实例会作为参数传到DataLoader中。它们用来指定数据加载中使用的indices/keys的顺序，它们是数据集索引上的可迭代对象。

  

  下面是正式的介绍

  简单来说，**sampler**是一个取样器（容器），用来对原始的数据取样，返回原始数据的多个子集，不同的类也对应不同的取样方式。DataLoader会根据参数中的`shuffle`参数自动构建一个sampler类实例，再传给DataLoader。若`shuffle`为True，即打乱数据，则参数`sampler` = `torch.utils.data.RandomSampler`；若为False，则`sampler` = `torch.utils.data.SequentialSampler`。

  在分布式训练时用到的是`distributed.DistributedSampler`。此种方法会根据当前分布式环境（具体说是worldsize）来将原始数据分为几个子集。

  **batch_sampler**的作用是从sampler中进行批处理，即将sampler中的数据分批，它返回的数据为一个batch的数据。具体细节将在下一小节讨论。

  **`distributed.DistributedSampler`参数**

  - **dataset** –要进行取样的数据集
  - **num_replicas** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 参与分布式训练的进程数量. `rank` 默认为当前进程组的进程数。
  - **rank** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) –当前进程在`num_replicas`的Rank，默认 `rank`从当前分布式组中检索。
  - **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True` (default), sampler 会打乱indices。
  - **seed** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 在 `shuffle=True`时，用来打乱采样器的随机种子，这个数字在分布式组中的所有进程之间应该是相同的Default: `0`。

  ***注意：在分布式模式下，在每个epoch开始之前应该调用 `sampler.set_eopch(i)`方法。*** 

- **batch_size、drop_last**以及**collate_fn**

  本小节与上一小节联系很大，建议联系到一起理解。

  `DataLoader`通过参数`batch_size`、`drop_last`和`batch_sampler`自动将获取的单个数据样本排序成批。

  如果`batch_size`（默认是1）的值不是`None`，数据加载器会生成成批的样本，每一批（batch）的样本数为batch_size的值。`drop_last`为True时，如果数据集size不能被batch size整除，会丢弃最后一个不完整的batch，此参数默认为False，也就是若不能整除，多出来的部分独占一个batch。若指定了 `batch_size`, `shuffle`, `sampler`和 `drop_last`中的任何一个（布尔值为True或具体指定）则`batch_sampler`就不能再指定了，因为会自动根据参数使用相应的类。

  `batch_size`和`drop_last`参数本质上是用来从sampler中构造batch_sampler的。对于map-style的数据集，sampler可以由用户提供，也可以基于`shuffle`参数构造，也就是上面说的，它们是互斥的。

  

  **collate_fn**在批处理和非处理是作用是不同的

  ​		若`batch_size`不是None，则为***自动成批模式***，此时使用`collate_fn`参数传递的函数来将一个列表中的样本排列为一个batch。（实际上，batch_sampler和sample作为取样器，返回的是根据规则排列的indices，并非真实的数据，还要使用`collate_fn`来排列真实数据）。 `collate_fn`每次调用一个列表里的数据样本，它需要将输入样本整理为批，以便从data loader迭代器生成。

  例如，如果每个数据样本由一个3通道图像和一个完整的类标签组成，也就是说数据集的每个元素都返回一个元组（`image，class_index`），默认的`collate_fn`会将包含这样的元组的列表整理成一个批处理过的图像tensor的单独的元组以及一个批处理过的类标签Tensor。具体来说，`collate_fn`有以下特点：

  - 它总是添加一个新维度作为批处理维度。
  - 它自动将NumPy数组和Python数值转换为PyTorch张量。
  - 它保留了数据结构，例如，如果每个样本是一个字典，它输出具有相同键集但批处理过的张量作为值的字典(如果值不能转换成张量，则值为列表)

  用户可以使用自定义的`collate_fn`来实现自定义批处理，例如沿第一个维度以外的维度排序、各种长度的填充序列或添加对自定义数据类型的支持。

  ​		当`batch_size`和`batch_sampler`都为None (`batch_sampler`的默认值已经为None)时，为***非自动成批模式***。此时使用作为`collate_fn`参数传递的函数来处理从数据集获得的每个示例。这时，这个函数只是将Numpy数组转换维PyTorch的Tensor，其他保持不变。

- **其他参数**

  **num_workers** 用来进行多进程加载数据，注意这里的多进程只是加载数据时的多进程，不同于多进程训练。在此模式下，每当创建一个`DataLoader`的迭代器时(例如，当调用`enumerate(dataLoader)`时)，会创建 `num_workers`个工作进程。此时，`dataset`，`collate_fn`和`worker_init_fn`被传你递给每个worker，它们被用于初始化和获取数据。这意味着数据集访问和它的内部IO，以及转换(包括collate_fn)都在工作进程中运行。

  也就是说只有对DataLoader迭代时才会得到真实的数据。

  **pin_memory** 为True 会自动将获取的数据张量放到固定的内存中，从而使数据更快地传输到支持cuda的gpu。

​	

以上就是在部署分布式训练需要了解的知识，更多细节参见官方文档。下面的配置流程为本教程的核心部分。



# 二 、使用过程框架

在DDP分布式训练中，关键是要在不同的进程中使用GPU进行数据处理，因此首先应该分配进程。假设只有一个机器，两块GPU。总数据量（视频或图片数量）为8000。batchsize设置为16。

​	准备工作：使用pytorch的spawn生成两个进程（对应GPU数量），分别使用1个GPU进行任务。在每个进程中都执行以下操作。

1. 初始化`torch.distributed`，这是DDP的依赖项。
2. 加载模型，如`model = model()`
3. 指定本进程对应的GPU：`torch.cuda.set_device(i)`  i 是当前进程对应的GPU号，以保证当前程在单独的GPU上运行
4. 将模型放到当前设备：`model.to(device)`
5. 模型并行化：`DistributedDataParallel(model,device_id=[i])`。
6. 数据处理，首先获取原始数据。
7. 根据分布式情况以及原始数据指定Sampler，作为DataLoader的参数输入。（将原始数据分为两个子集，每个子集有4000个副本）
8. 使用DataLoader包装原始数据，由于传入了Sampler，会使用batch_sampler 在sampler中再进行分批。由于使用了分布式，在此步之前将batch_size除以设备数，得到新的batch_size（8），作为每个GPU的batch_size。因此batch_sampler会根据batch_size和sampler产生$4000/8 = 500$ 个batch。
9. 在epoch中进行训练。注意，在每个epoch的开端调用`sampler.set_epoch(n)` n为epoch数。
10. 保存模型

# 三、代码解析

这部分将对应第二部分，给出每一步的代码以及详细的解释或明，但是作为分布式教程，下文主要针对与分布式相关的代码，而其他部分，如优化策略，学习率改变方法等不进行详细介绍。

本实验（图像分类）是在双显卡环境下进行的，在四块显卡的服务器上指定了0,3号显卡：`os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'`



​	首先分配进程

```python
import torch.multiprocessing as mp

opt.world_size = opt.ngpus_per_node * opt.world_size
mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
```

​	***代码说明：*** opt为整个程序用到的参数，batch_size，num_classes等参数都已指定，在下文中，每个参数出现时都会进行说明。这里的`opt.world_size`为总节点数（机器）,由于本教程针对单机多卡，因此设置为1。`opt.ngpus_per_node` 是每个节点的GPU数，设置为2，因此经过运算`opt.world_size`为2。`mp.spawn`产生了两个进程，每个进程都运行 main_worker函数（ main_worker是训练的主函数，包括模型、数据的加载，以及训练，以下所有内容都是在main_worker函数中的）

```python
def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
    dist.init_process_group(backend='nccl',
                            init_method=opt.dist_url,
                            world_size=opt.world_size,
                            rank=opt.dist_rank)
    opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
    opt.n_threads = int((opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model = make_data_parallel(model, opt.distributed, opt.device)
    parameters = model.parameters()
    criterion = CrossEntropyLoss().to(opt.device)

    (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)

    
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed)
            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)         
        scheduler.step()
        

```
#### 1. 初始化`torch.distributed`

```python
def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
    dist.init_process_group(backend='nccl',
                            init_method=opt.dist_url,
                            world_size=opt.world_size,
                            rank=opt.dist_rank)
    opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
    opt.n_threads = int((opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = opt.dist_rank == 0
```

***代码说明：*** 在每个进程中，都会分配一个`index`，由于我们有两个进程，所以在两个进程中的`index` 分别为0，1。同样的，opt为传入的参数，前三行代码为指定用到的随机seed。然后根据`index` 分别指定每个进程的device：cuda:0 和cuda:1（对应实际的0号和3号GPU）。接着指定`opt.dist_rank`，它将作为初始化时的rank参数，`opt.dist_rank`原始值为0，因此经过运算，在两个进程中的值分别为0，1。

下面就是本步的核心，初始化`torch.distributed`在它的参数里，在每个进程中`init_method`和`world_size`都是一样的，`rank`用来标识各自的进程，同样的，分别为0，1。

因为分了两个进程，所以对原始指定的`batch_size`，`n_threads`(DataLoader中的num_workers)除以进程数2。

#### 2. 加载模型

```python
 model = generate_model(opt)
```

此部分没什么好说的，从其他函数或类中获取模型。

但是注意到在它之后还有一段代码，是用来操作batch_norm的，这里不做过多解释，感兴趣可以查看原文档。

#### 3. 指定本进程对应的GPU

#### 4. 将模型放到当前设备

#### 5. 模型并行化

```python
model = make_data_parallel(model, opt.device)
    
def make_data_parallel(model, device):
   
    if device.type == 'cuda' and device.index is not None:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        model.to(device)

        model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
    
```

***代码说明：*** 在两个进程中分别对模型进行并行化，local_rank是获得每个进程的rank，分别为0，1。device在第一步中已经定义过。

三行代码分别对应三个步骤。

#### 6. 数据处理，获取原始数据

```python
train_data = get_training_data(**kwargs)
```

***代码说明：***根据参数获取原始数据

#### 7. 根据分布式情况以及原始数据指定Sampler，作为DataLoader的参数输入

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
```

#### 8. 使用DataLoader包装原始数据

```python
train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
```

#### 9. 在epoch中进行训练

```python
 for i in range(opt.begin_epoch, opt.n_epochs + 1):

    train_sampler.set_epoch(i)
    current_lr = get_lr(optimizer)
    train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed)
```





#### 以上即为本教程的全部内容，虽然没有涵盖训练的每个细节，但是你可以学会在你的代码中适当的位置添加某些内容，从而实现分布式训练。

#### 本教程仅为本人观点，如果有错误之处，欢迎评论！