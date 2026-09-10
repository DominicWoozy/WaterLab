# 专用计算内核实验（2026-09-10）

基线 `b99e3742494da3b870ae3ae0f31b654931e98bea`。这是前面论文建议中的第三项：按功能专门编译内核。本轮实现均保留为独立实验，没有接入网页，没有推送或发布。

## 依据与实现范围

[Bilotta 等：Fast, feature-rich weakly-compressible SPH on GPU: coding strategies and compiler choices](https://arxiv.org/abs/2207.11328)介绍重型计算内核的专门优化与编译器差异；相关 2024 年论文为 [Optimization of flexible neighbors lists in Smoothed Particle Hydrodynamics on GPU](https://www.sciencedirect.com/science/article/pii/S0965997824001182)。本轮重新读取了前者摘要；后者页面返回 403。这里借鉴内核专用化思路，不复现 CUDA 代码，也不把其加速比例套用到 WebGPU / Metal。

分别尝试：

1. **张力模式专用编译**：将粘度内核里的张力模式替换为编译期常量，生成关闭、显式、隐式三种版本。每次写参数时记录该参数缓冲对应的模式，在原有调度位置选择内核。
2. **精简资源绑定**：移除模块完全没有引用的缓冲变量声明，保留实际使用的绑定编号与访问类型。对流体和隐式张力内核应用此处理，检验减少无用绑定能否降低调度成本。没有移除实际读写。
3. **粒子档位与时间步常量化**：为 15k / 30k / 50k 编译对应版本，使用与现有 uniform 相同的 f32 尺度和固定物理时间步，让编译器提前计算部分表达式。额外覆盖速度准备内核。模式与档位切换选择匹配版本。

粒子数量、求解轮数、相互作用半径的参数定义、压力和张力公式、渲染配置均保持不变。常量折叠可能影响浮点舍入，因此验证数值误差，不宣称逐位相同。原型没有增加 GPU dispatch、回读或模拟缓冲；额外创建计算管线，有编译及管线存储成本。

## 性能

M3 Pro、原生 Metal、五万粒子、1250×800，反射、焦散和鸭子等保持开启。预热 120 次模拟，保存并恢复相同粒子、鸭子、反作用力和时间；每两帧交替版本，80 帧舍去前 20 帧。测第一物理 pass 到最终渲染结束的 GPU 时间戳，保持生产分段提交方式。

| 实验 / 场景                   |    通用版 |    专用版 | 判断                        |
| ----------------------------- | --------: | --------: | --------------------------- |
| 张力模式 / 平静               | 17.939 ms | 17.904 ms | 约 0.2%，不足以证明整帧收益 |
| 张力模式＋精简绑定 / 平静     | 18.009 ms | 18.007 ms | 持平                        |
| 张力模式＋精简绑定 / 大浪     | 19.082 ms | 19.093 ms | 持平                        |
| 再加尺度与时间步常量化 / 平静 | 17.990 ms | 17.957 ms | 约 0.2%，不足以证明整帧收益 |

只比较同一行，不使用不同运行的绝对时间计算收益。这不是浏览器 FPS，也不代表其他 GPU。没有对七万粒子测量。

固定五万粒子状态的分项测试中，每个测量 pass 连续执行 40 次粘度内核，交替四轮、舍去首轮：显式张力内核由约 0.221～0.222 ms 降至 0.212～0.213 ms，约快 3.7～4.7%；隐式模式中该内核的局部收益也约为 4%。每帧只执行三次，节约约 0.025～0.031 ms，约占 18 ms 完整区间的 0.2%。因此局部收益与整帧没有明显变化并不矛盾。隐式模式的这一数字只针对其粘度／阻尼内核，不是完整隐式求解器。

精简绑定没有显示出额外的整帧收益。不能据此断言驱动内部完全没有相关开销，因为未测硬件计数器或内部屏障。管线创建实测受驱动缓存影响明显，不能将一次编译时间当成稳定启动成本。

## 正确性

- 复用现有独立 CPU 全粒子邻居、压力、密度与粘度对照：稀疏、孤立、跨单元、512 个列表超过 96 邻居，以及缓存刷新均通过。
- 复用张力测试：吸引、短程排斥、关闭与隐式模式、粒子数量、线动量和角动量检查均通过。
- 同一输入逐项比较三种粒子质量档、稀疏／密集／孤立粒子与 `关闭→显式→隐式→显式→关闭` 切换。非档位常量化版的显式、隐式粘度输出逐项相同；关闭模式存在微小舍入差异。档位常量化版也通过 2e-6 的绝对容差要求，两版测量最大绝对误差均约为 2.98e-8。
- 全部成功运行没有 GPU 验证错误。未做浏览器视觉 QA，也未做七万档位或完整长期回归。

## 保留内容与结论

不接入运行代码：局部内核确有小幅收益，但增加管线与维护路径没有换来可感知的整帧改善。当前网页继续使用原来的实现。

保存了[测量数据](./experiments/specialized-kernel-results.json)、着色器原型、数值对照及基准脚本。从项目根目录、能访问 Metal GPU 的本地 Node 环境运行：

```sh
node --experimental-strip-types docs/experiments/specialized-kernel-correctness.mjs
FREEZE=1 node --experimental-strip-types docs/experiments/specialized-kernel-correctness.mjs
node --experimental-strip-types docs/experiments/specialized-kernel-benchmark.mjs
node --experimental-strip-types docs/experiments/specialized-kernel-frame-benchmark.mjs
PRUNE=1 SCENE=rough node --experimental-strip-types docs/experiments/specialized-kernel-frame-benchmark.mjs
PRUNE=1 FREEZE=1 node --experimental-strip-types docs/experiments/specialized-kernel-frame-benchmark.mjs
```
