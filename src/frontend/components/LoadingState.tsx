"use client";

import React from "react";
import { Loader2, Waves } from "lucide-react";

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = "Processing satellite patch and generating 15-depth temperature profile...",
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 bg-white/70 backdrop-blur-xs border border-slate-200 rounded-xl space-y-3 text-slate-600">
      <div className="relative flex items-center justify-center">
        <Waves className="w-8 h-8 text-cyan-500 animate-pulse" />
        <Loader2 className="w-12 h-12 text-cyan-600 animate-spin absolute" />
      </div>
      <p className="text-xs font-medium text-slate-700 tracking-wide text-center max-w-sm">
        {message}
      </p>
      <span className="text-[10px] text-slate-400 font-mono">
        3×3 CNN Encoder → Latent Space → MLP Decoder
      </span>
    </div>
  );
};
