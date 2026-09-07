# WATER · 粒子水体实验室

这次实现使用真实的三维粒子状态计算水流，不再通过固定矩形的波浪高度场制造运动。

## 运行与检查

```sh
npm install
npm run dev
npm run build
npx tsc --noEmit
node --test tests/fluid-simulation.test.mjs
```

## 交互

- **搅水**：按住拖动，对附近粒子施加推动和旋转力；轻点产生飞溅。
- **注水**：按住向对应位置持续注水，最多 2600 个粒子。水量 +/- 每次注入或移除 250 个粒子。
- **观察**：拖动旋转、滚轮缩放；任何模式下右键或 Shift + 拖动均可旋转。
- **晃动容器**：施加短暂的横向惯性力，引起撞壁、回流与飞溅。
- **参数**：搅动力度、黏性、重力、时间速度、光照、反射、水底光纹。
- **显示计算粒子**：切换为独立粒子视图，直接观察水体的离散计算单元。
- **键盘**：聚焦画布后，方向键旋转，+/- 缩放，回车或空格产生飞溅。

## 模拟

`app/fluid-simulation.ts` 保存每个粒子的位置与速度。在固定 1/120 秒的子步中施加重力和用户外力，通过空间网格查找相邻粒子，使用双密度松弛施加成对对称的位置修正，再处理容器边界碰撞、重建速度并施加相邻速度平滑黏性。

初始 1700 粒子从一列水柱释放，落下并在容器内扩散。平静/涌动/翻涌预设施加不同的整体外力；不是在渲染中移动纹理。

算法参考：[Clavet、Beaudoin 与 Poulin，Particle-based Viscoelastic Fluid Simulation (2005)](https://diglib.eg.org/items/7cad7994-b781-40ce-ad04-0bf61dc94279)。采用流体压力部分，没有实现论文的弹性弹簧。

## 渲染

`app/water-engine.ts` 使用原生 WebGL 2 绘制球形粒子深度，累积厚度，再通过多次深度双边平滑重建连续水面。`app/water-shaders.ts` 从平滑深度重建法线，计算 Fresnel 反射、折射、随厚度变化的吸光和程序化天空照明。

渲染方法参考：[Simon Green / NVIDIA，Screen Space Fluid Rendering for Games](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf)。

这是有限粒子分辨率下的交互流体近似。边界为固定容器，晃动通过惯性力近似；水面平滑、体积厚度、焦散和反射环境为实时图形近似，不等同于工程级水动力学仿真。粒子运动在 CPU 计算，表面重建及光学着色在 GPU 完成。需要 WebGL 2 和 EXT_color_buffer_float，帧率依设备而异。

可选 WebMCP 工具仅在支持 `document.modelContext` 的环境注册；当前环境没有提供该 API 的专门验证上下文。

## 验证范围

自动检查覆盖重力下落、零重力、粒子数量守恒、注水/排水上限、边界约束、搅水响应、重置、最大粒子量强扰动下的数值稳定性及非法时间步。浏览器交互与视觉效果未进行自动化测试。
