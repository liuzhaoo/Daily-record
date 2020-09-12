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