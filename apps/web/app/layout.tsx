import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YatraLink",
  description: "Find a way, not just a train.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
