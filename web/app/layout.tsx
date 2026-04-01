import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CAPA — Computational Architecture for Predicting Alloimmunity",
  description:
    "Structure-aware HLA mismatch representations for post-transplant outcome prediction.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
