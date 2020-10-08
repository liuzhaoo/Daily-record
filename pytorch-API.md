## TORCH

### Tensors

#### is_tensor

判断对象是否为tensor，相当于```is_instance(obj,Tensor)```,返回bool。用法：`is_tensor(obj)`

### Creation Ops

> Random sampling(随机采样)操作在 Random sampling 下，包括 `torch.rand()` `torch.rand_like()`  `torch.randn()` `torch.randn_like()` `torch.randint()`  `torch.randint_like()` `torch.randperm()`  

#### tensor

> `torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor`

​	使用数据来构造一个tensor,此方法会复制数据，如果输入数据是NumPy的`ndarray`,而想避免复制，可使用`torch.as_tensor`

​	如果数据x是tensor，则`torch.tensor(x)` 等价于`x.clone().detach()`,`torch.tensor(x,requires_grad=True)` 等价于				    	`x.clone().detach().requires_grad_(True)`  此时推荐使用后面的方法。 

​	**Parameters**  

​		*data*  为要转换的数据 ，可以是列表等数据形式

​		*dtype* 为数据类型，*device* 用来指定设备，*requires_grad* 表示是否需要计算梯度

#### as_tensor

> `torch.as_tensor(data, dtype=None, device=None) → Tensor`

​	将数据转换为`torch.Tensor` ，如果数据是tensor，而且与此方法指定的 *dtype* 和 *device* 相同，就不会复制tensor（此时应该只	是添加了一个新的指向）

#### from_numpy

> `torch.from_numpy(ndarray) → Tensor`

​	将`ndarray`类型的数据转换为tensor，返回的tensor使用原来数据的内存 。对 tensor 的修改会对原来数据产生影响，反之亦然。注意不可对返回的 tensor 做改变形状的操作

#### zeros/ones

> `torch.zeros`(**size*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

​	产生指定size的用0或1填充的tensor，一般只用第一个参数，其他参数默认

#### zeros_like/ones_like

> `torch.zeros_like`(*input*, *dtype=None*, *layout=None*, *device=None*, *requires_grad=False*, *memory_format=torch.preserve_format*) → Tensor

​	生成与`input`相同大小的0或1tensor（后面的各种参数也相等），等价于`torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)` 输入是Tensor

#### arange

> `torch.arange`(*start=0*, *end*, *step=1*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

返回一个一维的tensor（类似于一维列表），以 *start* 和 *end-1*  为端点，*step* 为步长，若不指定 *start* 则使用默认值0

**Example:**

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```

#### range

与`arrange` 类似，但是是以*start* 和 *end* 为端点

#### linspace

> `torch.linspace`(*start*, *end*, *steps=100*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

生成一维的tensor，其中 *steps* 是返回tensor的元素总数，默认为100，步长计算为$(end-start)\div(steps-1)$ 

**Example**

```python
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=1)
tensor([-10.])
```

#### logspace

> `torch.logspace`(*start*, *end*, *steps=100*, *base=10.0*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

生成`steps`size的一维tensor，起点是 $base^{start}$,终点是$base^{end}$ , 步长计算为$base^{(end-start)\div(steps-1)}$

#### eye

> `torch.eye`(*n*, *m=None*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

生成二维对角矩阵，n是行数，m为可选参数，用于指定列数，默认等于n‘

#### empty / empty_like

返回未初始化的tensor。`torch.empty((n,m))` `torch.empty_like(input)`  input是tensor

 #### quantize_per_tensor

> `torch.quantize_per_tensor`(*input*, *scale*, *zero_point*, *dtype*) → Tensor

将浮点型张量转换为给定scale和零点的量化张量。量化是指以低于浮点精度的位宽存储张量的技术，即可以减小模型尺寸，降低内存带宽要求，通常用于推理过程，因为不支持后向传播

 $$Q(x,scale,zero\_point) = round(\frac{x}{scale} + zero\_point)$$

### 索引，切片，连接，变异操作

#### cat

> `torch.cat`(*tensors*, *dim=0*, *out=None*) → Tensor

在指定的维度（必须是给出的tensor已有的维度）上对给出的tensor进行连接

**example**

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

#### chunk

>  `torch.chunk`(*input*, *chunks*, *dim=0*) → List of Tensors

cat 的反操作:在指定维度将tensor分为 chunks个tensor,若该维度的长度不能整除chunks,则最后一个取最小值.

## torch.nn

### DataParallel Layers (multi-GPU, distributed)

#### DataParallel



#### DistributedDataParallel





## torch.distributed

### Backends(后端)

`torch.distributed `支持三种内置后端，它们分别有不同的功能，下表显示哪些函数可用于CPU/CUDA张量。仅当用于构建PyTorch的实现支持时，MPI才支持CUDA。



​                      **Backend**                   |                    **gloo**                     |                    **mpi**                      |                     **nccl**  

|     Device     | CPU  | GPU  | CPU  | GPU  | CPU  | GPU  |
| :------------: | :--: | :--: | :--: | :--: | :--: | :--: |
|      send      |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✖   |
|      recv      |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✖   |
|   broadcast    |  ✔   |  ✔   |  ✔   |  ？  |  ✖   |  ✔   |
|   all_reduce   |  ✔   |  ✔   |  ✔   |  ？  |  ✖   |  ✔   |
|     reduce     |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✔   |
|   all_gather   |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✔   |
|     gather     |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✖   |
|    scatter     |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✖   |
| reduce_scatter |  ✖   |  ✖   |  ✖   |  ✖   |  ✖   |  ✔   |
|   all_to_all   |  ✖   |  ✖   |  ✔   |  ？  |  ✖   |  ✖   |
|    barrier     |  ✔   |  ✖   |  ✔   |  ？  |  ✖   |  ✔   |

PyTorch distributed目前只支持Linux。默认情况下，Gloo和NCCL后端是在PyTorch distributed中构建和包含的(只有在使用CUDA构建时才使用NCCL)。MPI是一个可选的后端，只有在从源代码构建PyTorch时才能包含它。(例如，在安装了MPI的主机上构建PyTorch。)

一般来说，使用GPU进行分布式训练时，使用NCCL后端

### 常用环境变量

- 选择要使用的网络接口

默认情况下，NCCL和Gloo后端都会尝试查找用于通信的网络接口。如果自动检测到的结构不正确，可以可以使用以下环境变量覆盖它(每个变量适用于其各自的后端）：

- **NCCL_SOCKET_IFNAME**, 比如 `export NCCL_SOCKET_IFNAME=eth0`
- **GLOO_SOCKET_IFNAME**, 比如 `export GLOO_SOCKET_IFNAME=eth0`

如果使用Gloo后端，可以指定多个接口，用逗号分隔它们：`export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3`后端将以循环方式跨这些接口分派操作。所有进程必须在此变量中指定相同数量的接口。

- 其他NCCL环境变量

- NCCL还提供了许多用于微调目的的环境变量

  常用的包括以下用于调试目的：

  - `export NCCL_DEBUG=INFO`
  - `export NCCL_DEBUG_SUBSYS=ALL`

  有关NCCL环境变量的完整列表，请参阅[NVIDIA NCCL的官方文档](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html)

### 基础

`torch.distributed`包为在一台或多台机器上运行的多个计算节点上的多进程并行结构提供PyTorch支持和通信原语。 [`torch.nn.parallel.DistributedDataParallel()`](https://pytorch.apachecn.org/docs/1.2/nn.html#torch.nn.parallel.DistributedDataParallel)类就是基于此功能构建的，作为任何PyTorch模型的包装来提供同步分布式训练。这不同于 [`Multiprocessing package - torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html) 和 [`torch.nn.DataParallel()`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) 提供的并行结构，因为它支持多台联网的机器而且用户必须显式地为每个进程启动主要训练脚本的副本。

在单机情况下， `torch.nn.parallel.DistributedDataParallel()`与其他数据并行方式相比，仍然具有优势：

- 每个进程都有其对应的优化器（optimizer）并且在每次迭代时都执行完整的优化步骤，虽然这看起来是多余的，但是因为各个进程之间的梯度已经收集到一起并平均，因此对于每个进程都是相同的，这意味着不需要参数广播步骤，减少了节点（GPU或主机）间传输张量的时间。
- 每个进程都包含一个独立的Python解释器，消除了额外的解释器开销和来自单个Python进程驱动多个执行线程，模型副本或GPU的“GIL-thrashing”。这对于大量使用Python运行时的模型尤其重要，包括具有循环层或许多小组件的模型

### 初始化

在调用其他任何方法之前，需要用`torch.distributed.init_process_group()` 对此包进行初始化，这将阻止所有进程加入。

> `torch.distributed.is_available`()

若返回 True 则证明分布式包可以使用。目前，`torch.distributed`支持Linux和Macos。当从源码构建Pytorch时设置`USE_DISTRIBUTED=1` Linux的默认值为1，Macos的默认值为0

> `torch.distributed.init_process_group`(*backend*, *init_method=None*, *timeout=datetime.timedelta(0*, *1800)*, *world_size=-1*, *rank=-1*, *store=None*, *group_name=''*)

初始化默认的分布式进程组，这也将同时初始化分布式包

​			**初始化进程组的方式有两种**

​				1. 明确指定`store`,`rank`,以及`world_size`

​				2. 指定`init_method`(一个URL字符串)，它指示在哪里/如何发现对等点,可以选择指定rank和world_size，或者在URL中					编码所有必需的参数并省略它们。

​			**如果两者都没有指定，则假设`init_method`为' 'env://''。**

​	**参数**

​		***backend***（字符串或后端端口名称），用到的后端。取决于构建时的配置，有效值包括`mpi`,`gloo`,`nccl` ，应该为小写的形		式，也可以通过后端属性访问，比如`Backend.GLOO` 。如果在每台机器上通过`nccl`后端来使用多个进程，每个进程必须独占		访问它使用的每个GPU，因为进程之间共享GPU会导致锁死。

​		***init_method*** (字符串，可选)，指定如何初始化流程组的URL。如果没有指定`init_method`或`store`，默认为"env://"。与
​		`store`相互排斥

​		***world_size*** (整数，可选)， 参与工作的进程数，若指定了`store`，则此项是必须的。

​		***rank***（整数，可选），当前进程的排名，若指定了`store`，则此项是必须的。

​		***store*** （存储，可选），所有任务都可访问的键值对，用来交换连接/地址信息。与`init_method`互斥

​		***timout***（时间间隔对象，可选），针对进程组执行的操作超时，默认值等于30分钟，这仅适用于`gloo`后端对于。`nccl`，只		有在环境变量`NCCL_BLOCKING_WAIT`被设置为1时才适用。

​		***group_name*** (字符串，可选，已弃用) 组名称。

> `CLASS`  torch.distributed.Backend

可用后端的类似于枚举的类：GLOO,NCCL,MPI，以及其他注册的后端。

这个类的值时小写字符串，比如"gloo"。可以将它们看作属性来进行访问：`Backend.NCCL`

可以直接使用此类来解析字符串，比如，`Backend(backend_str)` 会检查`backend_str`是否有效，如果是，会返回解析后的小写字符串。也可以接受大写字符串。`Backend("GLOO")`返回"`gloo`"

> `torch.distributed.get_backend`(*group= < object object>*)

返回给定进程组的后端

> `torch.distributed.get_backend`(*group= < object object>*)

返回当前进程组的rank

Rank是分配给分布式进程组中的每个进程的唯一标识符。它们是从0到world_size的连续整数。

>  `torch.distributed.get_backend`(*group= < object object>*)

返回当前进程组中的进程数

> `torch.distributed.is_initialized`()

检查默认进程组是否已初始化

> `torch.distributed.is_nccl_available`()

检查NCCL后端是否可用





**目前支持三种初始化方式**

### TCP 初始化

使用TCP进行初始化的方法有两种，都需要一个所有进程都可以访问的网络地址和一个设定好的的`world_size`。

第一种方法需要指定一个属于rank0进程的地址。此初始化方法要求所有进程都手动指定rank。

注意，在最新的分布式包中不再支持多播地址。也不赞成使用`group_name`。

```python
import torch.distributed as dist

# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)
```

### 共享文件初始化

另一种初始化方法使用一个文件系统，该文件系统与所需的`world_size` 都可以被同组组中的所有机器共享并可见。URL应以`file：//`开头，并包含共享文件系统上不存在的文件(在现有目录中）的路径。如果文件不存在，文件系统初始化将自动创建该文件，但不会删除该文件。因此，下一次在相同的文件路径中初始化 [`init_process_group()`](https://pytorch.apachecn.org/docs/1.2/distributed.html#torch.distributed.init_process_group) 之前，应该确保已经清理了文件。

```python
import torch.distributed as dist

# rank should always be specified
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)
```

### 环境变量初始化

此方法将从环境变量中读取配置，从而可以完全自定义信息的获取方式。要设置的变量是：

- `MASTER_PORT` - 需要;，必须是机器上的rank为0的空闲端口，。
- `MASTER_ADDR` - 需要，(0级除外）; rank 0节点的地址。
- `WORLD_SIZE` - 需要，可以在这里设置，也可以在调用init函数时设置。
- `RANK` - 需要，可以在这里设置，也可以在调用init函数时设置。

等级为0的机器将用于设置所有连接。

这是默认方法，意味着不必指定`init_method`(或者可以是`env：//`）。

### GROUPS 组

默认情况下，集合在默认组（也叫world）上运行，并要求所有进程进入分布式函数调用。然而，一些工作负载可以从更具细粒度的通信中受益，这就是分布式组发挥作用的地方。`new_group` 函数可以用来创建一个新的组，，这个组具有所有进程的任意子集。它返回一个不透明的组句柄，可以将其作为`group`的参数提供给所有集合（在某些众所周知的编程模式中，集合是交换信息的分布式函数）

> `torch.distributed.new_group`(*ranks=None*, *timeout=datetime.timedelta(0*, *1800)*, *backend=None*)

创建一个新的分布式组

此函数要求主组中的所有进程(即属于分布式作业的所有进程）都进入此函数，即使它们不是该组（新建的组）的成员。此外，应在所有进程中以相同的顺序创建组。

### 点对点（P2P）通信

> `torch.distributed.``send`(*tensor*, *dst*, *group=<object object>*, *tag=0*)

同步发送张量

​	**参数:**

- **tensor** ([*Tensor*](https://pytorch.apachecn.org/docs/1.2/tensors.html#torch.Tensor)) – 准备发送的张量。
- **dst** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 发送的目标的rank。
- **group** (*ProcessGroup__,* *optional*) – 要处理的进程组。
- **tag** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 用来匹配发送与远程接收的tag。

> `torch.distributed.recv`(tensor, src=None, group=<object object>, tag=0)

同步接收张量

**参数：**

- **tensor** ([*Tensor*](https://pytorch.apachecn.org/docs/1.2/tensors.html#torch.Tensor)) – 接收到的数据转换为张量。
- **src** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 指定接受的来源rank。如果未指定，将接受任何进程的数据。
- **group** (*ProcessGroup__,* *optional*) – 要处理的进程组。
- **tag** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 用来匹配发送与远程接收的tag。

[`isend()`](https://pytorch.apachecn.org/docs/1.2/distributed.html#torch.distributed.isend) 和 [`irecv()`](https://pytorch.apachecn.org/docs/1.2/distributed.html#torch.distributed.irecv) 使用时返回分布式请求对象。通常，此对象的类型未指定，因为它们永远不应手动创建，但它们保证支持两种方法：

- `is_completed()` - 如果操作已完成，则返回True。
- `wait()` - 将阻止该过程，直到操作完成，`is_completed(）`保证一旦返回就返回True。

### 同步和异步的集体操作

每个集体操作函数都支持以下两种操作：

同步操作——当`async_op`被设置为False时的默认模式。当函数返回时，保证执行集体操作(如果它是CUDA操作，则不一定完成，因为所有CUDA操作都是异步的），并且可以调用任何进一步的函数调用，这取决于集体操作的数据。在同步模式下，集体函数不返回任何内容。

当`async_op`被设置为True时为异步步操作，集合操作函数返回一个分布式请求对象。一般来说，你不需要手动创建它，它保证支持两种方法:

- `is_completed()` - 如果操作已完成，则返回True。
- `wait()` - 将阻止该过程，直到操作完成。

### 集体函数

> `torch.distributed.broadcast`(*tensor*, *src*, *group=<object object>*, *async_op=False*)

将tensor广播到所有的组中

在所有参与集合的进程中，tensor的元素数量都应保持一致

**参数:**

- **tensor** ([*Tensor*](https://pytorch.apachecn.org/docs/1.2/tensors.html#torch.Tensor)) – 如果`src`是当前进程的rank，则为发送的数据，否则用于保存接收数据的张量。
- **src** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 来源rank。
- **group** (*ProcessGroup__,* *optional*) – 要处理的进程组。
- **async_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 这个操作是否应该是异步操作。

**返回值**

​	若async_op设置为True，则返回句柄，否则为None

> `torch.distributed.all_reduce`(*tensor*, *op=ReduceOp.SUM*, *group=<object object>*, *async_op=False*)

减少所有机器上的张量数据，以便获得最终结果，调用之后，`tensor`在所有进程中都会按位相同。

**参数:**

- **tensor** ([*Tensor*](https://pytorch.apachecn.org/docs/1.2/tensors.html#torch.Tensor)) – 集合的输入和输出。该功能就地运行。
- **op** (*optional*) – 来自`torch.distributed.ReduceOp`枚举的值之一。指定用于逐元素减少的操作。
- **group** (*ProcessGroup__,* *optional*) – 要处理的进程组。
- **async_op** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 这个操作是否是异步操作。









## Distributed Overview

pytorch中，torch.distributed 的特性主要可以分为三个部分:

- [Distributed Data-Parallel Training](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) 是一种广泛采 用的单程序多数据训练范例，在每个进程上复制模型，每个模型副本都输入不同的数据样本。DDP注重于梯度通信，以保持模型副本之间的同步，同时将梯度与梯度计算重叠来加速训练。

  相关api： `torch.nn.DataParallel`，`torch.nn.parallel.DistributedDataParallel`

  ##### DataParallel

  可以进行单机器多GPU的设置，只需一行代码，但是此方法性能不是最好的，因为其实现方法是在每个前向传播中都重复模型，它的单进程多线程并行结构自然会受到GIL竞争的影响。

  ##### DistributedDataParallel

  相比于 `DataParallel`，`DistributedDataParallel`需要更多步骤来进行设置，比如调用`init_process_group`，DDP使用多进程并行结构，因此模型副本之间不存在GIL争用。此外，模型在构建DDP时广播（到各进程），而不是在每一次前向传递时广播，这也有助于加快训练速度

  

- [RPC-Based Distributed Training](https://pytorch.org/docs/master/rpc.html) (RPC，Remote Procedure Call) 概括来说就是不同机器训练同一个模型，使用一个一个更高级的API来自动区分在多台机器上分布的模型

  相关api： [torch.distributed.rpc](https://pytorch.org/docs/master/rpc.html)

  此方法没遇到过

- [Collective Communication](https://pytorch.org/docs/stable/distributed.html) (c10d) 库支持在一组中跨进程发送张量，它支持集体通信API（如`torch.distributed.all_reduce`和`torch.distributed.all_gather`）和P2P通信API（如`torch.distributed.send`和`torch.distributed.isend`）。DDP和RPC（[ProcessGroup Backend](https://pytorch.org/docs/master/rpc.html#process-group-backend)）在1.6.0版本中的c10d上建立，前者使用集体通信，后者使用P2P通信。通常，开发人员不需要直接使用这个原始通信API（c10d），因为上面的DDP和RPC特性可以用于许多分布式训练场景

**数据并行训练**

pytorch为数据并行训练提供了很多方法，对于从简单到复杂、从原型到生产的应用程序，通常的开发顺序是:

1. 如果GPU能够容纳模型和数据，而且不考虑训练速度，使用单设备训练。
2. 若有多个GPU并且想修改少量代码就可以加速训练，使用单个机器多GPU（`torch.nn.DataParallel`）
3. 若想更好地加速训练，可以使用单机器多GPU`torch.nn.parallel.DistributedDataParallel`的方法，但是需要写更多代码
4. 若需要在多个机器上使用多个GPU，可以使用`torch.nn.parallel.DistributedDataParallel`和` launching script`
5. 使用`torchelastic` 来进行分布式训练（如果错误可以预测或者在训练过程中可以动态地添加或减少资源）



