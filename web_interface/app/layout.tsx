import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Gaia Network Demo',
  description: 'A demonstration of the Gaia Network system for integrated climate, financial, and risk analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 min-h-screen`}>
        {children}
      </body>
    </html>
  )
} 