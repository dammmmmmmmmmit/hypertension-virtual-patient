import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { SiteNav } from "@/components/site-nav";

// Display face for headlines — Space Grotesk has real geometric character
// without being gimmicky, and reads as "technical/data-native" rather than
// decorative, which fits a pharmacology screening tool.
const spaceGrotesk = Space_Grotesk({
  variable: "--font-heading",
  subsets: ["latin"],
});

// Clean, highly legible body face.
const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

// Anything that's genuinely data — dosages, mmHg values, dataset rows,
// model metrics — renders in mono. Grounded in the subject matter, not
// decorative (ui_build_brief.md).
const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Virtual Patient Drug-Response Simulator",
  description:
    "In-silico hypertension drug-combination screening — real pharmacology data, a fine-tuned local LLM, and a trained prediction pipeline.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground font-sans">
        <SiteNav />
        {children}
      </body>
    </html>
  );
}
