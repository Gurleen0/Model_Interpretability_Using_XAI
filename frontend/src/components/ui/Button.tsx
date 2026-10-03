import React from "react";

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "outline";
  disabled?: boolean;
  className?: string;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = "primary",
  disabled = false,
  className = "",
}) => {
  const baseClasses =
    "px-6 py-3 rounded-xl font-medium transition-all duration-300 transform active:scale-95 focus:outline-none focus:ring-2 focus:ring-purple-600 focus:ring-offset-2 focus:ring-offset-gray-900";
  const variantClasses = {
    primary: disabled
      ? "bg-gray-700 text-gray-500 cursor-not-allowed"
      : "bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:opacity-90",
    secondary: disabled
      ? "bg-gray-800 text-gray-500 cursor-not-allowed"
      : "bg-gray-800 text-white hover:bg-gray-700",
    outline: disabled
      ? "border border-gray-700 text-gray-500 cursor-not-allowed"
      : "border border-gray-700 text-gray-300 hover:border-purple-600 hover:text-white",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
    >
      {children}
    </button>
  );
};
