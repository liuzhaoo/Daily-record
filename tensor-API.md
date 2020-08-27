## TORCH

### Tensors

#### is_tensor

判断对象是否为tensor，相当于```is_instance(obj,Tensor)```,返回bool。用法：`is_tensor(obj)`

### Creation Ops

> Random sampling(随机采样)操作在 Random sampling 下，包括 `torch.rand()` `torch.rand_like()`  `torch.randn()` `torch.randn_like()` `torch.randint()`  `torch.randint_like()` `torch.randperm()`  

#### tensor

使用数据来构造一个tensor

> `torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor`



