import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = "" }) => {
  return (
    <div
      className={`bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-lg ${className}`}
    >
      {children}
    </div>
  );
};
