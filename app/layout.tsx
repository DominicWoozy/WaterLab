import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {
  title: 'WATER — 水体实验室',
  description:
    '真正由粒子驱动的水体实验室。拖动搅水、持续注水、晃动容器，探索重力、黏性与动态反射。',
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN" className="dark">
      <body>{children}</body>
    </html>
  );
}
