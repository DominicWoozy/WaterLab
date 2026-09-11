# 多界面水体折射的加速论文筛选

检索日期：2026-09-11。已发布的折射基线为 `ca54567dafb59e1ccb51e9776d028e75f094a7ce`；本笔记记录之后的研究，没有接入新的渲染算法。

目标是在保留粒子物理、密度场、界面折射、厚度吸收和图像分辨率的前提下减少光线计算。当前成本基线见 `exit-refraction.md`：50k / 1250×800 的水体绘制阶段，平静约 3.53 ms，浪花约 6.41 ms。整帧不能按水体绘制阶段的加速比例等比例提升。

下列“应用到 WaterLab”均为结合代码的工程判断，未测得加速结果。论文的 CPU/GPU、数据分布和比较基线与本项目不同，不搬用其中的加速倍数。

## 1. 优先：保守 min/max 加速结构 + 网格 DDA

论文依据：Wald 等，2005，[Faster Isosurface Ray Tracing using Implicit KD-Trees](https://www.sci.utah.edu/~wald/Publications/2005/iso/tcgv.pdf)。第 3.3、4.1 节在层级节点保存密度最小/最大值；目标等值不在区间内，就跳过整个子树。论文实现主要面向 CPU / SIMD，不能把其性能直接作为 WebGPU 预期。

遍历基础：Amanatides、Woo，1987，[A Fast Voxel Traversal Algorithm for Ray Tracing](https://www.eecs.yorku.ca/~amana/research/grid.pdf)。按光线跨越网格边界的先后增量遍历，不需要沿整条射线均匀取大量点。

应用到 WaterLab：优先做浅层规则宏块，而非直接照搬复杂 KD-tree。每帧在最终密度纹理生成后，计算宏块的保守 min/max；光线在块中时可以区分三种情况：

- `max < iso`：整块空气，只推进位置。
- `min > iso`：整块水，只推进位置并累计水中长度。
- 区间包含 iso：仍使用现有密度采样、交点细化和法线。

三线性插值是单元八个角值的凸组合，因此使用全部相关角值的 min/max 可以保守排除界面。这种应用不以降低几何分辨率换速度。但必须包含跨宏块边界的插值支撑节点，不能只读块中心；当前 `density()` 的容器裁切和动态顶部也必须纳入边界处理。浮点范围要保守，相等/接近阈值的情况回退精查。

其他必须保留的语义：水内跳跃仍累计吸收长度；先用最近不透明交点限制射线区间；每次折射/反射后重新初始化 DDA。解析水滴单独处理，不能因为密度宏块为空就误判解析水滴为空。

按每轴 4 个插值单元分块估算，127×159×95 个单元需要 32×40×24 个宏块，两个 f32 范围值为 245,760 字节（约 0.234 MiB）；浅层层级再增加少量存储。这仅是设计估算，不是实测总显存。真正要比较的是每帧构建/更新成本与减少的纹理采样成本。第一轮保留表面附近的步进与法线策略，单独检查采样起点变化是否造成图像差异。

## 2. 次优先：缓存单元系数，隔离最近交点后迭代求根

Marmitt 等，2004，[Fast and Accurate Ray-Voxel Intersection Techniques for Iso-Surface Ray Tracing](https://www.sci.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf)。三线性密度沿单元内直线是三次多项式。论文第 4 节先用极值分隔根，再从前向后定位第一个含根区间，并迭代细化；后续计算复用多项式系数，减少重复插值。它还指出，只比较线段两端密度会漏掉两次穿越。

应用到 WaterLab：只对 min/max 判断可能包含表面的单元读取八个角值并建立系数；用算术求值替代重复纹理采样。可同时减少重采样并改进小空隙/薄层交点检测。它没有创造密度网格中原本不存在的细节。

风险：硬件三线性采样本身很快，建立系数、求极值和分支可能更贵，尤其 M3 的硬件过滤路径。论文约三倍结果比较的是当时 CPU 上不同精确求交函数，不是本项目整帧。保留当前平滑法线；改成单元解析梯度会改变表面观感，应作为独立实验。

## 3. 后续：把复杂光线分阶段、压紧队列

Laine、Karras、Aila，2013，[Megakernels Considered Harmful: Wavefront Path Tracing on GPUs](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)。用多个专用内核组织光线工作，降低大内核的控制流分歧和寄存器压力。

应用到 WaterLab：将穿越密度场、界面事件和不透明物体查询拆开，压紧仍活跃的光线，尤其针对多次全反射或多层喷溅。当前这些逻辑集中在片元着色器中。

调度本身可以保留数学模型，但会增加队列读写、扫描/原子操作和 dispatch；当前只有有限次界面事件与少量材质，未必值得。需要先测活跃光线分布和长尾工作量，不宜第一步重构整条渲染管线。

## 4. 可直接参考 WebGPU 代码：2023 年渐进光线调度

Usher、Dyken、Kumar，LDAV 2023，[Speculative Progressive Raycasting for Memory Constrained Isosurface Visualization of Massive Volumes](https://www.willusher.io/publications/wgpu-prog-iso/)，[作者的 WebGPU 实现](https://github.com/Twinklebear/webgpu-prog-iso)。算法在 GPU 上逐步遍历光线、按需解压体数据，并通过推测性处理更多射线块来利用空闲并行度。

适合借鉴队列、分块和 GPU 工作分配。不宜直接使用跨帧逐步完成图像的呈现方式：本项目水体每帧变化，需要避免不同时间状态的水面混合。我们也没有大体数据按需解压需求。论文的内存/解压量降幅不是 FPS 提升，不能作为本项目收益预测。

## 其他候选及适用边界

[Acceleration Techniques for GPU-based Volume Rendering (Krüger、Westermann，2003)](https://www.cs.cit.tum.de/cg/research/publications/2003/acceleration-techniques-for-gpu-based-volume-rendering/) 是 GPU 空区域跳跃、提前终止的经典依据。但不能直接按“不透明度足够”提前截断水的透射；本项目是界面折射与水程吸收，需要保留后方接收面。

[SparseLeap (Hadwiger 等，2018)](https://vcg.seas.harvard.edu/publications/20180101-sparseleap-efficient-empty-space-skipping-for-large-scale-volume-rendering) 先用光栅化生成每像素有效光线区间，降低光线阶段的空间查询成本。其区间对应原来的直线；光线出入水后方向改变，不能原样复用。动态水也要承担更新边界与区间的成本，故不作为这轮多次折射的首选。

## 建议实验顺序

1. 新增宏块 min/max，并用 DDA 跳过确定不含界面的空间；原场景、薄片、水滴和遮挡全部保留。
2. 同时测缓存构建、主水面绘制和整帧 GPU 时间；固定相同物理状态交替比较。增加近距离、多层水及全反射场景，覆盖有/无硬件浮点过滤。
3. 用独立高精度光线解核对最近交点、介质状态、出射方向、水中长度与最近遮挡；关注宏块边缘、零方向分量、容器裁切和细空气缝。不能只比较平均帧率。
4. 如果精查仍占主要成本，再测单元系数求交；仅当分歧长尾明显且收益超过队列成本时考虑 wavefront。

本轮只研究，没有承诺具体收益，也未自动接入或发布这些候选算法。
