import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'WATER — 水体实验室', description: '探索水的体积、波浪与光。可交互的 WebGL 水体模拟，支持实时反射、折射、焦散和点击涟漪。' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="zh-CN" className="dark"><body>{children}</body></html>; }
