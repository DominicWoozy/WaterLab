# 保持物理预算的逐项优化验证

2026-10-16，基线 `4ebc5aa056ab5be8ae9e632b009a5b04dffdb73a`。验证阶段仅在本地运行；高光与色彩原型不进入网站默认渲染路径。

## 1. 高粒子数分桶：采用

5 万以下保持原尺度。7 万 / 10 万的 cell size 改为 `0.225 × particleScale`，网格扩展到 `36×45×27` / `41×51×31`，覆盖原来的空间范围。前缀和支持非整块末尾和 254 个扫描块，缓存编号仍为 u32，分配最大网格并按当前档位清空、扫描。

物理核半径、粒子质量、每 tick 的 3 个子步、每子步的压力迭代次数（高档 6 次）、张力、黏性、浮力和重建核均未调整。cell size 大于 `h + 0.042 × scale`，仍容纳两次重建之间的一次最大压力/接触位移。改变排序会改变浮点求和次序，不能宣称长时间轨迹逐位一致；验证的是完整邻居集合、独立数值参考和既有稳定性指标。

M3 Pro，原生 Metal WebGPU，1250×800，硬件浮点过滤；默认张力、反射、焦散、精细主粒子水滴开启，自反射/自阴影关闭。预热 120 tick；恢复同一份粒子、鸭子、反作用和时间，对照新旧模拟/重建。120 帧舍弃前 20 帧，每方案 50 帧。GPU 时间戳均值：

| 粒子 / 场景 | 物理阶段：旧 → 新 | 整帧 GPU：旧 → 新 |    整帧耗时变化 |
| ----------- | ----------------: | ----------------: | --------------: |
| 5 万，平静  |  13.46 → 13.60 ms |  20.68 → 20.92 ms | +1.2%，基本持平 |
| 7 万，平静  |  27.83 → 24.62 ms |  35.91 → 32.22 ms |          −10.3% |
| 10 万，平静 |  48.74 → 36.89 ms |  57.85 → 45.49 ms |          −21.4% |
| 10 万，扰动 |  47.11 → 35.79 ms |  60.02 → 47.76 ms |          −20.4% |

只比较每行内部配对结果；不同场景的频率、温度和水形不同。整帧指物理起点到水体绘制结束，不含浏览器合成，也不能当成 FPS。增加的网格和扫描存储约 0.39 MiB。

15 秒模拟后的高档静水窗口：

| 粒子  | RMS 压缩误差 | RMS 速度 | 每 tick RMS 位移 |   平均浮力 / 重量 |
| ----- | -----------: | -------: | ---------------: | ----------------: |
| 7 万  |       0.693% |  0.00992 |         0.000165 | 0.25507 / 0.25480 |
| 10 万 |       0.885% |  0.01114 |         0.000186 | 0.25506 / 0.25480 |

均通过原有 1% 压缩、抖动和浮力平衡门槛；没有放宽断言。

## 2. 世界空间焦散复用：采用

引擎每次重建水体后增加 lighting revision；渲染器缓存水体、鸭子、光照和模型对应的焦散图及鸭子照度。只移动相机、调整显示尺寸、画鼠标圈或等待下一物理 tick 时可复用。reset、drain、物理更新、精细水滴切换、退出粒子调试等引起重建时，下一次照明立即重新求解。光照 / 焦散开关和模型替换也使缓存失效。独立调用者不提供 revision 时始终重新求解。

没有降低光子数、光线路径预算、焦散分辨率或物理更新频率；没有新增图像副本。运动中的物理帧没有这项收益。

`tests/webgpu-lighting-cache.mjs` 使用波浪密度场和真实鸭子网格，对照缓存和强制重算。初帧、相机、鼠标圈、反射、光强、焦散开关、鸭子位姿、清空水体和模型重载均检查失效行为；两种采样路径实际最大像素差均为 **0**（测试允许 1 个量化级，以容纳光子原子顺序）。另有引擎调度测试检查 revision。

640×400 静态场景、单帧提交并等待完成的交替测量：

| 密度采样   | 显示帧 GPU：重算 → 复用 | CPU 提交至 GPU 完成：重算 → 复用 |
| ---------- | ----------------------: | -------------------------------: |
| 硬件过滤   |        1.114 → 0.719 ms |                 1.789 → 1.250 ms |
| 手动三线性 |        1.948 → 1.405 ms |                 2.526 → 1.916 ms |

此处为固定场景的显示帧，不能解释为整个动态模拟提速 35%。较早的双帧在途测量无法稳定隔离阶段，未用作收益结论。

## 3. 高光抗锯齿：完成实验，暂不采用

参考 [Filtering Distributions of Normals for Shading Antialiasing, HPG 2016](https://research.nvidia.com/publication/2016-06_filtering-distributions-normals-shading-antialiasing)。当前水面是经验性 Phong 日光高光，并非论文里的 GGX / Beckmann 微表面模型，不能把简单加粗糙度等同于完整论文实现。

实际原型：保持命中点、密度、法线与折射路径不变，在统一控制流取得相邻像素的反射方向差分，对原有日光高光做四点像素内积分。无时间历史；未混合上一帧。它只验证主水面日光高光，未处理所有环境反射、解析小水滴或二次水面反射，因此不接入生产。

同一份 5 万粒子扰动后冻结状态，512×320，使用每轴 4 倍分辨率（16 样本/输出像素）的原始高光作为参考；仅统计有效高光像素的线性强度误差：

| 视角 | 原始 RMS 误差 | 原型 RMS 误差 | 参考 / 原型高光能量 |
| ---- | ------------: | ------------: | ------------------: |
| 俯视 |       0.06069 |       0.05231 |     157.03 / 159.93 |
| 侧视 |       0.05812 |       0.04980 |     118.59 / 125.52 |
| 远景 |       0.07990 |       0.05298 |       63.03 / 65.50 |

误差降低约 14%–34%，但侧视仍多约 5.8% 能量，不能称为准确积分。完整水体绘制阶段均值 2.219 → 2.190 ms，差异在测量波动范围内，只支持“当前样例无明显开销”，不支持加速结论。未完成运动序列的时间稳定性验收。后续应统一微表面/光源模型并处理轮廓处差分，才考虑默认启用。

## 4. 统一色彩输出：完成实验，暂不采用

[Filament 的线性色彩与成像说明](https://google.github.io/filament/main/filament.html) 要求光照在线性空间运算，最终经过显示转换。代码中的鸭子贴图目前用 `pow(2.2)` 解码，而最终用了经验曲线 `pow(0.91)`。

实验将贴图改为精确 sRGB 解码，最终输出改为标准 sRGB 编码。没有改变物理、吸收或曝光。完整图像平均 RGB 绝对差约 **0.22**（0–1 范围），画面整体明显变亮；绘制成本约 2.203 ms，与基线相当。原有环境色、材质和光强已经按旧输出调过，因此只替换编码不能保证原有观感，也不是已经完成的 PBR 校准。暂保留当前默认颜色。

左侧默认图、实验高光图及 sRGB 输出图分别保存在 `experiments/appearance-images/`；数值见 `experiments/appearance-results.json`。后续色彩改进需要一并校准光源、环境、材质和曝光，而不是为追求“清透”修改水的密度/吸收。

## 验证与复现

通过：TypeScript、展示/调度测试、Sites 构建、GitHub Pages 构建及静态资源路径检查；本地浏览器 WebGPU 10 万档位、扰动、暂停/恢复、焦散切换及无控制台错误检查；原生 GPU 的高档初始化/完整排序/边界覆盖、50k/70k/100k 全粒子数值参考和缓存溢出回退、高档长期静水、50k 扰动数值参考、全部档位重建复用、墙面贴合、焦散缓存两种过滤方式、自光学开关图像测试。

```sh
# 基线目录保留 app/ 下相对依赖；性能实验需要原生 Metal。
mkdir -p /tmp/water-grid-baseline
git archive 4ebc5aa056ab5be8ae9e632b009a5b04dffdb73a app | tar -x -C /tmp/water-grid-baseline
BASELINE_ROOT=/tmp/water-grid-baseline/app node --experimental-strip-types docs/experiments/adaptive-grid-benchmark.mjs 100000
SCENE=rough BASELINE_ROOT=/tmp/water-grid-baseline/app node --experimental-strip-types docs/experiments/adaptive-grid-benchmark.mjs 100000
HIGH_QUALITY=1 node --experimental-strip-types tests/webgpu-settling.mjs
QUALITY=70000 node --experimental-strip-types tests/webgpu-neighbors.mjs
QUALITY=100000 node --experimental-strip-types tests/webgpu-neighbors.mjs
node --experimental-strip-types tests/webgpu-lighting-cache.mjs
UNFILTERED=1 node --experimental-strip-types tests/webgpu-lighting-cache.mjs
node --experimental-strip-types docs/experiments/appearance-validation.mjs
```

外观原型仅位于实验目录，不会打包进网站。实验每次生成的新图片存于 `/tmp/water-appearance-validation`。
