### 内核融合第三部分：线性层的 UOps

在我之前关于内核融合和调度项的概述中，我讨论了生成内核时的三个抽象层次：`ScheduleItem`、`UOps` 以及最终的内核代码，并详细解释了 `ScheduleItem` 的生成过程。接下来，我们来讨论如何将 `ScheduleItem` 转换为所有需要执行的操作的线性表示（即 `UOps`），以及这些操作的优化过程。

回顾一下，点积操作会生成如下的 `ScheduleItem`，之后会被转换为内核代码：

```plaintext
  0 ━┳ STORE MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)))
  1  ┗━┳ SUM (0,)
  2    ┗━┳ MUL
  3      ┣━━ LOAD MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)))
  4      ┗━━ LOAD MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)))
```

上述的 AST 树结构就是我们之前讨论的 `ScheduleItem`，接下来它会被转换为以下线性表示：

```plaintext
step  Op_name               type                      input                           arg
   0  UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (0, 'data0', True)
   1  UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (1, 'data1', False)
   2  UOps.DEFINE_GLOBAL  : ptr.dtypes.int            []                               (2, 'data2', False)
   3  UOps.DEFINE_ACC     : dtypes.int                []                               0
   4  UOps.CONST          : dtypes.int                []                               0
   5  UOps.CONST          : dtypes.int                []                               2
   6  UOps.LOOP           : dtypes.int                [4, 5]                           None
   7  UOps.LOAD           : dtypes.int                [1, 6]                           None
   8  UOps.LOAD           : dtypes.int                [2, 6]                           None
   9  UOps.ALU            : dtypes.int                [7, 8]                           BinaryOps.MUL
  10  UOps.ALU            : dtypes.int                [9, 3]                           BinaryOps.ADD
  11  UOps.PHI            : dtypes.int                [3, 10, 6]                       None
  12  UOps.ENDLOOP        :                           [6]                              None
  13  UOps.STORE          :                           [0, 4, 11]                       None
```

你可以参考之前的高层次概述以获取更多上下文信息。

#### `corealize` 方法的起点

```python
@staticmethod
def corealize(lst: Iterable[Tensor]):
    run_schedule(create_schedule(flatten([x.lazydata.lbs if isinstance(x.lazydata, MultiLazyBuffer) else [x.lazydata] for x in lst])))
```

我们会生成一个 `ScheduleItem`（即本文讨论的内容），然后调用 `run_schedule` 函数，该函数会在内部调用 `prg = lower_schedule_item(si)`，如果该项包含顶层的 `STORE` 操作，它将通过 `get_runner` 方法开始将 `ScheduleItem` 转换为 `UOps` 的过程。

```python
if si.ast[0].op is BufferOps.STORE: 
    return Device[si.outputs[0].device].get_runner(*si.ast)
```

在 `get_runner` 内，它会初始化一个 `Linearizer` 并调用 `to_program` 方法（*ast 是我们的 `ScheduleItem`）：

```python
def get_runner(self, *ast: LazyOp) -> CompiledASTRunner: 
    return self.to_program(self.get_linearizer(*ast))
```

`Linearizer` 类负责将 AST（`ScheduleItem`）转换为线性操作，因此得名。接下来，我们进入 `linearize()` 方法：

```python
def linearize(self):
```

这是一个非常长的方法，关键在于 `self.uops` 属性，它保存了所有 `UOps`（微操作）的列表。通过观察何时和如何将项添加到这个列表中，我们可以理解这个过程是如何运作的。

`self.uops` 在这里被初始化：

```python
self.uops: UOpGraph = UOpGraph()
```

然后通过 `self.uops.add` 方法来添加 `UOps`，例如：

```python
self.buf_uops[i] = self.uops.add(UOps.DEFINE_GLOBAL, buf.dtype if isinstance(buf.dtype, ImageDType) else PtrDType(buf.dtype), (), (buf.idx, f"data{buf.idx}", any(buf.idx == x.idx for x in self.outbufs)))
```

`add` 方法的实现如下：

```python
def add(self, uop: UOps, dtype: Optional[DType] = None, vin: Tuple[UOp, ...] = tuple(), arg: Any = None, cachable=True, insert_before=None, simplify=True) -> UOp:
    ret = UOp(uop, dtype, vin, arg)
    ...
    self.uops.insert(insert_before, ret)
    return ret
```

接着，进入一个典型的循环展开和累积器设置部分。在 `reduce` 操作（如 `SUM`）中，会设置累积器，并通过循环将多个元素归约为一个。

整个过程最终会生成优化的 `UOps` 列表，经过进一步的优化步骤后，生成可执行的内核代码。

这就是如何将 `ScheduleItem` 转换为线性操作（`UOps`）的过程。 [oai_citation:1,tinygrad_notes_o1.md](file-service://file-975CFde1oHeMm5wnHNfOM4qR)
