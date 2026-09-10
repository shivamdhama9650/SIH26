"use client";

import React from "react";
import { AlertCircle, RefreshCw, X } from "lucide-react";

interface ErrorMessageProps {
  message: string | null;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  message,
  onRetry,
  onDismiss,
}) => {
  if (!message) return null;

  return (
    <div className="bg-red-50/90 border border-red-200 rounded-xl p-4 shadow-sm text-red-800 flex items-start justify-between space-x-3 transition-all">
      <div className="flex items-start space-x-3">
        <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-semibold text-red-900">
            Inference / Service Notification
          </h4>
          <p className="text-xs text-red-700 mt-0.5 leading-relaxed">
            {message}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2.5 inline-flex items-center space-x-1 px-2.5 py-1 bg-red-100 hover:bg-red-200 text-red-800 rounded text-xs font-medium transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry Request</span>
            </button>
          )}
        </div>
      </div>

      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-700 p-1 rounded transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};
