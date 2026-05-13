import { I18nManager } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Localization from 'expo-localization';
import { I18n } from 'i18n-js';
import { translations } from './translations';

const STORAGE_KEY = 'bovitech.language';

export const i18n = new I18n(translations);
i18n.enableFallback = true;
i18n.defaultLocale = 'fr';

function normalizeLocale(loc) {
  if (!loc) return 'fr';
  const base = String(loc).split('-')[0].toLowerCase();
  return base === 'ar' ? 'ar' : 'fr';
}

export function isRtlLocale(locale) {
  return normalizeLocale(locale) === 'ar';
}

export async function initI18n() {
  const saved = await AsyncStorage.getItem(STORAGE_KEY);
  const device = Localization.getLocales?.()?.[0]?.languageTag;
  const locale = normalizeLocale(saved || device || Localization.locale);
  i18n.locale = locale;

  // RTL (arabe). Remarque: un changement RTL nécessite parfois un redémarrage complet.
  const shouldBeRtl = isRtlLocale(locale);
  if (I18nManager.isRTL !== shouldBeRtl) {
    I18nManager.allowRTL(true);
    I18nManager.forceRTL(shouldBeRtl);
  }

  return locale;
}

export async function setLanguage(locale) {
  const normalized = normalizeLocale(locale);
  await AsyncStorage.setItem(STORAGE_KEY, normalized);
  i18n.locale = normalized;

  const shouldBeRtl = isRtlLocale(normalized);
  if (I18nManager.isRTL !== shouldBeRtl) {
    I18nManager.allowRTL(true);
    I18nManager.forceRTL(shouldBeRtl);
  }
  return normalized;
}

export function getLanguage() {
  return i18n.locale;
}

export function t(key, options) {
  return i18n.t(key, options);
}

