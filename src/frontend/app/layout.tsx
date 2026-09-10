import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OceanXRay | North Indian Ocean Temperature Profile Prediction",
  description:
    "Scientific research platform predicting 15-depth ocean temperature profiles from satellite-derived surface patches over the North Indian Ocean.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
