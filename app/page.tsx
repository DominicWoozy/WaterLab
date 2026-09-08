'use client';
import { useEffect, useRef, useState } from 'react';
import {
  Waves,
  Droplets,
  Sun,
  RotateCcw,
  Pause,
  Play,
  MousePointer2,
  Move,
  ZoomIn,
  SlidersHorizontal,
  Plus,
  Minus,
  Orbit,
  Hand,
  Activity,
} from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { registerWaterTools } from './webmcp';
import {
  createWater,
  defaults,
  type WaterSettings,
  type WaterStats,
} from './water-engine';
import {
  GPU_CAPACITY as CAPACITY,
  GPU_DEFAULT_COUNT as DEFAULT_COUNT,
  type ParticleQuality,
} from './gpu-particle-config';
function Range({
  label,
  value,
  min,
  max,
  step = 0.01,
  suffix = '',
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="range-field">
      <div className="field-label">
        <span>{label}</span>
        <output>
          {value.toFixed(2)}
          <span>{suffix}</span>
        </output>
      </div>
      <Slider
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
      />
    </div>
  );
}
export default function Home() {
  const [settings, setSettings] = useState<WaterSettings>(defaults),
    [stats, setStats] = useState<WaterStats>({ fps: 0, count: DEFAULT_COUNT }),
    [quality, setQuality] = useState<ParticleQuality>(DEFAULT_COUNT),
    [preset, setPreset] = useState('平静'),
    [error, setError] = useState(''),
    [ready, setReady] = useState(false),
    [panelOpen, setPanelOpen] = useState(true);
  const canvas = useRef<HTMLCanvasElement>(null),
    engine = useRef<ReturnType<typeof createWater> | null>(null),
    current = useRef(settings);
  current.current = settings;
  useEffect(() => {
    try {
      engine.current = createWater(
        canvas.current!,
        () => current.current,
        (next) => {
          setStats(next);
          if (next.quality) setQuality(next.quality);
        },
        setError,
      );
      setReady(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : '无法启动水体渲染');
    }
    return () => engine.current?.destroy();
  }, []);
  useEffect(
    () =>
      registerWaterTools(
        () => current.current,
        (next) => {
          current.current = next;
          setSettings(next);
          setPreset('自定义');
        },
        () => engine.current?.ripple(),
      ),
    [],
  );
  const update = <K extends keyof WaterSettings>(
    key: K,
    value: WaterSettings[K],
  ) => setSettings((s) => ({ ...s, [key]: value }));
  const choose = (name: string) => {
    setPreset(name);
    setSettings((s) => ({
      ...s,
      agitation: name === '平静' ? 0 : name === '涌动' ? 0.4 : 1.4,
      viscosity: 0.025,
      paused: false,
    }));
    if (name === '翻涌') engine.current?.shake();
  };
  const reset = () => {
    setSettings(defaults);
    setPreset('平静');
    engine.current?.reset();
  };
  const act = (action: 'ripple' | 'shake' | 'pour' | 'drain') => {
    if (!engine.current) return;
    if (action !== 'drain') update('paused', false);
    engine.current[action]();
  };
  const modeText =
    settings.mode === 'stir'
      ? '按住水面拖动，推动水流；轻点让水跃起。'
      : settings.mode === 'pour'
        ? '按住任意位置，向容器中持续注入水流。'
        : '拖动旋转视角，滚轮缩放；也可用方向键观察。';
  return (
    <main className="water-app particle-app">
      <canvas
        ref={canvas}
        className="water-canvas"
        tabIndex={0}
        aria-label="粒子流体。默认拖动搅水，点击飞溅。观察模式或右键拖动旋转，滚轮缩放。方向键旋转，加减键缩放，回车扰动。"
      />
      <header className="topbar">
        <a className="brand" href="/" aria-label="WATER 水体实验室">
          <span className="brand-icon">
            <Waves size={24} />
          </span>
          <span>
            WATER<span className="brand-dot">.</span>
          </span>
          <span className="brand-divider" />
          <span className="brand-subtitle">水体实验室</span>
        </a>
        <div className="header-right">
          <span className="live-dot" />
          <span>粒子流体</span>
          <span className="version">{stats.backend || 'GPU 初始化'}</span>
        </div>
      </header>
      <section className="scene-title">
        <div className="eyebrow">
          <span /> PARTICLE FLUID STUDY
        </div>
        <h1>推、搅、倾注。</h1>
        <p>{modeText}</p>
      </section>
      <button
        className="panel-toggle icon-button"
        aria-label={panelOpen ? '收起控制面板' : '展开控制面板'}
        onClick={() => setPanelOpen(!panelOpen)}
      >
        <SlidersHorizontal size={18} />
      </button>
      {panelOpen && (
        <aside className="control-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">EXPERIMENT 012</span>
              <h2>自由水流</h2>
            </div>
            <Activity size={19} />
          </div>
          <Tabs defaultValue="water">
            <TabsList className="control-tabs">
              <TabsTrigger value="water">
                <Droplets size={15} />
                水体
              </TabsTrigger>
              <TabsTrigger value="light">
                <Sun size={15} />
                光照
              </TabsTrigger>
            </TabsList>
            <TabsContent value="water">
              <div className="section-label">流动状态</div>
              <p
                style={{
                  fontSize: '0.875rem',
                  color: 'var(--muted-foreground)',
                  margin: '0 0 12px',
                }}
              >
                搅动水面，观察小鸭子随浪漂浮。
                <a
                  href="https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Duck"
                  target="_blank"
                  rel="noreferrer"
                  style={{ textDecoration: 'underline' }}
                >
                  模型 © Sony
                </a>
              </p>
              <div className="presets">
                {['平静', '涌动', '翻涌'].map((p, i) => (
                  <button
                    key={p}
                    aria-pressed={preset === p}
                    className={preset === p ? 'selected' : ''}
                    onClick={() => choose(p)}
                  >
                    <svg viewBox="0 0 40 22" aria-hidden="true">
                      <path
                        d={
                          i === 0
                            ? 'M2 12 Q12 10 20 12 T38 12'
                            : i === 1
                              ? 'M2 12 Q7 3 12 12 T22 12 T32 12 T38 12'
                              : 'M2 17 Q6 -2 11 12 T21 12 T31 12 T38 6'
                        }
                      />
                    </svg>
                    {p}
                  </button>
                ))}
              </div>
              <Range
                label="搅动力度"
                value={settings.strength}
                min={0.2}
                max={2.5}
                suffix="×"
                onChange={(v) => update('strength', v)}
              />
              <Range
                label="黏性"
                value={settings.viscosity}
                min={0}
                max={1}
                onChange={(v) => {
                  update('viscosity', v);
                  setPreset('自定义');
                }}
              />
              <Range
                label="重力"
                value={settings.gravity}
                min={0}
                max={14}
                step={0.1}
                suffix="m/s²"
                onChange={(v) => update('gravity', v)}
              />
              <Range
                label="时间速度"
                value={settings.speed}
                min={0.25}
                max={1.5}
                suffix="×"
                onChange={(v) => update('speed', v)}
              />
              <div className="quality-control">
                <span className="section-label">模拟精度</span>
                <Tabs
                  value={String(quality)}
                  onValueChange={(v) => {
                    const next = Number(v) as ParticleQuality;
                    setQuality(next);
                    engine.current?.setQuality(next);
                    setStats((s) => ({ ...s, count: next }));
                  }}
                >
                  <TabsList aria-label="粒子精度">
                    <TabsTrigger value="15000" disabled={!ready || !!error}>
                      15,000
                    </TabsTrigger>
                    <TabsTrigger value="30000" disabled={!ready || !!error}>
                      30,000
                    </TabsTrigger>
                    <TabsTrigger
                      value="50000"
                      disabled={!ready || !!error || stats.backend !== 'WebGPU'}
                    >
                      50,000
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
                <p>
                  切换会重置水体。5 万粒子使用 WebGPU；不支持时自动回退 WebGL2。
                </p>
              </div>
              <div className="particle-count">
                <div>
                  <span>水量</span>
                  <output>
                    {stats.count.toLocaleString()}
                    <small>
                      {' '}
                      / {(stats.capacity ?? CAPACITY).toLocaleString()} 粒子
                    </small>
                  </output>
                </div>
                <div className="quantity-buttons">
                  <button
                    aria-label="减少 500 个水粒子"
                    disabled={!ready || !!error || stats.count === 0}
                    onClick={() => act('drain')}
                  >
                    <Minus size={15} />
                  </button>
                  <button
                    aria-label="注入 500 个水粒子"
                    disabled={
                      !ready ||
                      !!error ||
                      stats.count >= (stats.capacity ?? CAPACITY)
                    }
                    onClick={() => act('pour')}
                  >
                    <Plus size={15} />
                  </button>
                </div>
              </div>
              <div className="particle-toggle">
                <label htmlFor="particles">查看物理粒子（调试）</label>
                <Switch
                  id="particles"
                  checked={settings.particles}
                  onCheckedChange={(v) => update('particles', v)}
                />
              </div>
            </TabsContent>
            <TabsContent value="light">
              <div className="section-label">自然天光</div>
              <div className="light-preview">
                <Sun size={28} />
                <span>
                  柔和日光<small>三维水面 · 按厚度吸光</small>
                </span>
              </div>
              <Range
                label="光照强度"
                value={settings.light}
                min={0.2}
                max={2.5}
                suffix="×"
                onChange={(v) => update('light', v)}
              />
              <div className="toggle-row">
                <label htmlFor="reflection">
                  环境反射<small>随粒子水面形变的环境倒影</small>
                </label>
                <Switch
                  id="reflection"
                  checked={settings.reflection}
                  onCheckedChange={(v) => update('reflection', v)}
                />
              </div>
              <div className="toggle-row">
                <label htmlFor="caustics">
                  水底光纹<small>动态焦散近似</small>
                </label>
                <Switch
                  id="caustics"
                  checked={settings.caustics}
                  onCheckedChange={(v) => update('caustics', v)}
                />
              </div>
              <p className="light-note">
                粒子运动与水体重建均在 GPU
                计算。薄水透明，深水随光程逐渐呈青蓝色。
              </p>
            </TabsContent>
          </Tabs>
          <div className="panel-actions">
            <button
              className="pause-button"
              onClick={() => update('paused', !settings.paused)}
            >
              {settings.paused ? <Play size={15} /> : <Pause size={15} />}{' '}
              {settings.paused ? '继续模拟' : '暂停模拟'}
            </button>
            <button
              className="reset-button"
              aria-label="重置水体、参数和视角"
              title="重置"
              onClick={reset}
            >
              <RotateCcw size={16} />
            </button>
          </div>
        </aside>
      )}
      <div className="scene-caption">
        <span className="caption-line" />
        <span>{settings.particles ? '计算粒子视图' : '连续水面视图'}</span>
        <span className="caption-detail">重力 · 压力 · 黏性 · 碰撞</span>
      </div>
      <div className="fluid-toolbar">
        <Tabs
          value={settings.mode}
          onValueChange={(value) =>
            update('mode', value as WaterSettings['mode'])
          }
        >
          <TabsList className="tool-modes">
            <TabsTrigger value="stir">
              <Hand size={16} />
              搅水
            </TabsTrigger>
            <TabsTrigger value="pour">
              <Droplets size={16} />
              注水
            </TabsTrigger>
            <TabsTrigger value="orbit">
              <Orbit size={16} />
              观察
            </TabsTrigger>
          </TabsList>
        </Tabs>
        <span className="toolbar-divider" />
        <button
          className="shake-button"
          disabled={!ready || !!error}
          onClick={() => act('shake')}
        >
          <Move size={16} />
          晃动容器
        </button>
      </div>
      {settings.paused && (
        <div className="paused-hint">已暂停 · 点击继续模拟恢复水流</div>
      )}
      <footer className="bottom-bar">
        <div className="interaction-hints">
          <span>
            <MousePointer2 size={14} /> 拖动搅水
          </span>
          <span>
            <Move size={14} /> 右键旋转
          </span>
          <span>
            <ZoomIn size={14} /> 滚轮缩放
          </span>
        </div>
        <div className="render-status">
          <span
            className={settings.paused ? 'status-dot paused' : 'status-dot'}
          />
          <span>
            {error
              ? '渲染不可用'
              : !ready
                ? '准备水体'
                : settings.paused
                  ? '已暂停'
                  : '模拟中'}
          </span>
          <span className="fps">{stats.fps || '—'} FPS</span>
        </div>
      </footer>
      {error && (
        <div className="error-message" role="alert">
          <h2>水体暂时无法显示</h2>
          <p>{error}</p>
          <button onClick={() => location.reload()}>重新加载</button>
        </div>
      )}
    </main>
  );
}
