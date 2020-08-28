## argparse

添加命令行参数，用法：

```python
import argparse

parser = argparse.ArgumentParser(description="ddd") 
parser.add_argument("aaa", type=str, default='bbb', help='ccc') # 参数前面没有加--表示是必须手动指定值的参数，并且不能使用默认值
# etc.al
args = parser.parse_args()
```

再用到相关参数时直接 `args.aaa` 就可以获取它的`default` 值。

- 注意，在参数前面加上 `--`  就可以在程序中指定该参数的缺省值，否则只能在命令行手动添加。