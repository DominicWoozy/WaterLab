'use client';
import { useEffect, useRef, useState } from 'react';
import { Waves, Droplets, Sun, RotateCcw, Pause, Play, MousePointer2, Move, ZoomIn, SlidersHorizontal, ArrowUpRight, Plus } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { registerWaterTools } from './webmcp';
import { createWater, defaults, type WaterSettings } from './water-engine';

function Range({ label, value, min, max, step = .01, suffix = '', onChange }: { label: string; value: number; min: number; max: number; step?: number; suffix?: string; onChange: (v: number) => void }) {
 return <div className="range-field"><div className="field-label"><span>{label}</span><output>{value.toFixed(2)}<span>{suffix}</span></output></div><Slider aria-label={label} min={min} max={max} step={step} value={[value]} onValueChange={v=>onChange(Array.isArray(v)?v[0]:v)} /><div className="range-ends"><span>{min.toFixed(1)}</span><span>{max.toFixed(1)}</span></div></div>;
}
export default function Home(){
 const [settings,setSettings]=useState<WaterSettings>(defaults),[preset,setPreset]=useState('涟漪'),[fps,setFps]=useState(0),[error,setError]=useState(''),[ready,setReady]=useState(false),[panelOpen,setPanelOpen]=useState(true);
 const canvas=useRef<HTMLCanvasElement>(null), engine=useRef<ReturnType<typeof createWater>|null>(null), current=useRef(settings);current.current=settings;
 useEffect(()=>{try{engine.current=createWater(canvas.current!,()=>current.current,setFps,setError);setReady(true);}catch(e){setError(e instanceof Error?e.message:'无法启动水体渲染');}return()=>engine.current?.destroy();},[]);
 useEffect(()=>registerWaterTools(()=>current.current,next=>{current.current=next;setSettings(next);setPreset('自定义');},()=>engine.current?.ripple()),[]);
 const update=<K extends keyof WaterSettings>(key:K,value:WaterSettings[K])=>setSettings(s=>({...s,[key]:value}));
 const choose=(name:string)=>{setPreset(name);setSettings(s=>({...s,amplitude:name==='静水'?.08:name==='涟漪'?.55:1.4,speed:name==='静水'?.35:name==='涟漪'?.8:1.5}));};
 const reset=()=>{setSettings(defaults);setPreset('涟漪');engine.current?.reset();};
 return <main className="water-app">
  <canvas ref={canvas} className="water-canvas" tabIndex={0} aria-label="交互式三维水体。拖动旋转，滚轮缩放，点击水面产生涟漪。键盘方向键旋转，加减键缩放，回车产生涟漪。" />
  <header className="topbar"><a className="brand" href="/" aria-label="WATER 水体实验室"><span className="brand-icon"><Waves size={24}/></span><span>WATER<span className="brand-dot">.</span></span><span className="brand-divider"/><span className="brand-subtitle">水体实验室</span></a><div className="header-right"><span className="live-dot"/><span>实时渲染</span><span className="version">WEBGL / 01</span></div></header>
  <section className="scene-title"><div className="eyebrow"><span/> INTERACTIVE WATER STUDY</div><h1>一方水，自在流动。</h1><p>光穿过水面，每一道波纹都有回应。</p></section>
  <button className="panel-toggle icon-button" aria-label={panelOpen?'收起控制面板':'展开控制面板'} onClick={()=>setPanelOpen(!panelOpen)}><SlidersHorizontal size={18}/></button>
  {panelOpen&&<aside className="control-panel"><div className="panel-heading"><div><span className="panel-kicker">EXPERIMENT 001</span><h2>水体控制</h2></div><SlidersHorizontal size={18}/></div>
  <Tabs defaultValue="water"><TabsList className="control-tabs"><TabsTrigger value="water"><Droplets size={15}/>水体</TabsTrigger><TabsTrigger value="light"><Sun size={15}/>光照</TabsTrigger></TabsList>
   <TabsContent value="water"><div className="section-label">波浪状态</div><div className="presets">{['静水','涟漪','风浪'].map((p,i)=><button key={p} aria-pressed={preset===p} className={preset===p?'selected':''} onClick={()=>choose(p)}><svg viewBox="0 0 40 22" aria-hidden="true"><path d={i===0?'M2 12 Q12 10 20 12 T38 12':i===1?'M2 12 Q7 3 12 12 T22 12 T32 12 T38 12':'M2 17 Q6 -2 11 12 T21 12 T31 12 T38 6'}/></svg>{p}</button>)}</div>
   <Range label="波浪强度" value={settings.amplitude} min={0} max={2} onChange={v=>{update('amplitude',v);setPreset('自定义');}}/>
   <Range label="流动速度" value={settings.speed} min={.1} max={2} suffix="×" onChange={v=>{update('speed',v);setPreset('自定义');}}/>
   <Range label="水体深度" value={settings.depth} min={.3} max={1.8} suffix="m" onChange={v=>update('depth',v)}/>
   <div className="material"><span className="color-swatch"/><div><span>澄澈青</span><small>清水 · 折射率 1.333</small></div><span className="material-code">H₂O</span></div>
   </TabsContent>
   <TabsContent value="light"><div className="section-label">自然天光</div><div className="light-preview"><Sun size={28}/><span>柔和日光<small>天空环境 · 实时高光</small></span></div><Range label="光照强度" value={settings.light} min={.2} max={2.5} suffix="×" onChange={v=>update('light',v)}/><div className="toggle-row"><label htmlFor="reflection">环境反射<small>随观察角度变化的水面倒影</small></label><Switch id="reflection" checked={settings.reflection} onCheckedChange={v=>update('reflection',v)}/></div><div className="toggle-row"><label htmlFor="caustics">水底焦散<small>水波聚光形成的流动光纹</small></label><Switch id="caustics" checked={settings.caustics} onCheckedChange={v=>update('caustics',v)}/></div><p className="light-note">旋转视角，观察光线在水面上的反射与水体中的衰减。</p></TabsContent>
  </Tabs>
  <div className="panel-actions"><button className="pause-button" onClick={()=>update('paused',!settings.paused)}>{settings.paused?<Play size={15}/>:<Pause size={15}/>} {settings.paused?'继续模拟':'暂停模拟'}</button><button className="reset-button" aria-label="重置所有参数和视角" title="重置" onClick={reset}><RotateCcw size={16}/></button></div>
  </aside>}
  <div className="scene-caption"><span className="caption-line"/><span>体积水体</span><span className="caption-detail">3.7 × 2.7 × {settings.depth.toFixed(2)} m</span></div>
  <div className="scene-controls"><button className="ripple-button" onClick={()=>engine.current?.ripple()} disabled={!ready||!!error}><Plus size={17}/> 添加涟漪 <ArrowUpRight size={14}/></button><span>也可以轻触水面</span></div>
  <footer className="bottom-bar"><div className="interaction-hints"><span><MousePointer2 size={14}/> 点击扰动</span><span><Move size={14}/> 拖动旋转</span><span><ZoomIn size={14}/> 滚轮缩放</span></div><div className="render-status"><span className={settings.paused?'status-dot paused':'status-dot'}/><span>{error?'渲染不可用':!ready?'准备水体':settings.paused?'已暂停':'模拟中'}</span><span className="fps">{fps||'—'} FPS</span></div></footer>
  {error&&<div className="error-message" role="alert"><h2>水体暂时无法显示</h2><p>{error}</p><button onClick={()=>location.reload()}>重新加载</button></div>}
 </main>;
}
