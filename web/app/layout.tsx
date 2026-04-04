import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "CAPA — Computational Architecture for Predicting Alloimmunity",
    template: "%s · CAPA",
  },
  description:
    "Structure-aware HLA mismatch representations for post-transplant outcome prediction using protein language models and deep competing-risks survival analysis.",
  keywords: ["HLA", "transplant", "alloimmunity", "GvHD", "survival analysis", "ESM-2", "deep learning"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen flex flex-col font-sans">
        <Navbar />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
