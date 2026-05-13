// src/constants/theme.js — aligné sur src/theme/colors.js (dashboard & stack)
export const COLORS = {
  primary: '#1B4332',
  primaryLight: 'rgba(255, 255, 255, 0.22)',
  accent: '#C99026',
  accentLight: 'rgba(255, 255, 255, 0.82)',
  background: '#F7F5F0',
  surface: '#FFFFFF',
  surfaceAlt: '#EEF5F0',
  danger: '#B84A4A',
  warning: '#B57A1C',
  success: '#40916C',
  info: '#2C5282',
  textPrimary: '#1E1E1C',
  textSecondary: '#5C5C57',
  textLight: '#8A8972',
  border: '#E8E4DB',
  white: '#FFFFFF',
};

export const FONTS = {
  regular: 'System',
  medium: 'System',
  bold: 'System',
  sizes: {
    xs: 11,
    sm: 13,
    md: 15,
    lg: 18,
    xl: 21,
    xxl: 25,
    xxxl: 32,
  },
};

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const RADIUS = {
  sm: 10,
  md: 16,
  lg: 22,
  xl: 26,
  full: 999,
};

export const SHADOWS = {
  card: {
    shadowColor: '#1B4332',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.06,
    shadowRadius: 24,
    elevation: 4,
  },
  strong: {
    shadowColor: '#1B4332',
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.1,
    shadowRadius: 30,
    elevation: 8,
  },
};
