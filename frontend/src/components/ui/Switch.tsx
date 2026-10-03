import React from "react";

interface SwitchProps {
  enabled?: boolean;
  onChange?: (enabled: boolean) => void;
  disabled?: boolean;
}

export const Switch: React.FC<SwitchProps> = ({
  enabled = false,
  onChange,
  disabled = false,
}) => {
  const handleClick = () => {
    if (!disabled && onChange) {
      onChange(!enabled);
    }
  };
  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors duration-300 ${
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"
      } ${
        enabled
          ? "bg-gradient-to-r from-purple-600 to-indigo-600"
          : "bg-gray-600"
      }`}
      role="switch"
      aria-checked={enabled}
    >
      <span
        className={`inline-block h-5 w-5 transform rounded-full bg-white shadow-lg transition-transform duration-300 ${
          enabled ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
};
