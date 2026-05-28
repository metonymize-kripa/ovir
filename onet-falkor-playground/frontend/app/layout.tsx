import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'O*NET × FalkorDB — Career Transition Engine',
  description: 'Structural competency gap analysis via GraphBLAS traversal',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
