/**
 * Météo via Open-Meteo (gratuit, sans clé API).
 * @see https://open-meteo.com/en/docs
 */

/** Centre Tunis — secours si géolocalisation refusée ou indisponible */
export const DEFAULT_FARM_COORDS = { latitude: 36.8065, longitude: 10.1815 };

const OPEN_METEO_BASE = 'https://api.open-meteo.com/v1/forecast';

/**
 * Rose des vents (8 directions), libellés FR / AR.
 */
function degreesToWindDir8(deg, lang) {
  const d = ((deg % 360) + 360) % 360;
  const idx = Math.round(d / 45) % 8;
  if (lang === 'ar') {
    const ar = ['ش', 'ش.ق', 'ق', 'ج.ق', 'ج', 'ج.غ', 'غ', 'ش.غ'];
    return ar[idx];
  }
  const fr = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
  return fr[idx];
}

/**
 * Codes météo WMO (Open-Meteo) → résumé court + emoji.
 * @see https://open-meteo.com/en/docs
 */
function wmoToSummary(code, lang) {
  const isAr = lang === 'ar';
  if (code === 0) return isAr ? { text: 'سماء صافية', icon: '☀️' } : { text: 'Ciel dégagé', icon: '☀️' };
  if (code === 1) return isAr ? { text: 'صافٍ غالباً', icon: '🌤️' } : { text: 'Principalement dégagé', icon: '🌤️' };
  if (code === 2) return isAr ? { text: 'غائم جزئياً', icon: '⛅' } : { text: 'Partiellement nuageux', icon: '⛅' };
  if (code === 3) return isAr ? { text: 'غائم', icon: '☁️' } : { text: 'Couvert', icon: '☁️' };
  if (code === 45 || code === 48) return isAr ? { text: 'ضباب', icon: '🌫️' } : { text: 'Brouillard', icon: '🌫️' };
  if (code >= 51 && code <= 57) return isAr ? { text: 'رذاذ', icon: '🌦️' } : { text: 'Bruine', icon: '🌦️' };
  if (code >= 61 && code <= 67) return isAr ? { text: 'مطر', icon: '🌧️' } : { text: 'Pluie', icon: '🌧️' };
  if (code >= 71 && code <= 77) return isAr ? { text: 'ثلج', icon: '❄️' } : { text: 'Neige', icon: '❄️' };
  if (code >= 80 && code <= 82) return isAr ? { text: 'زخات مطر', icon: '🌦️' } : { text: 'Averses', icon: '🌦️' };
  if (code >= 95) return isAr ? { text: 'عواصف رعدية', icon: '⛈️' } : { text: 'Orages', icon: '⛈️' };
  return isAr ? { text: 'متغيّر', icon: '🌤️' } : { text: 'Variable', icon: '🌤️' };
}

function pastureHintFromConditions(code, tempC, lang) {
  const isAr = lang === 'ar';
  const rain = (code >= 61 && code <= 67) || (code >= 80 && code <= 82) || code >= 95;
  const hot = tempC >= 32;
  const cold = tempC <= 2;

  if (rain) {
    return isAr
      ? 'هطول مطر متوقع — راقب ظروف المرعى والراحة للأبقار.'
      : 'Précipitations possibles : adaptez la sortie au pâturage et le confort du troupeau.';
  }
  if (hot) {
    return isAr
      ? 'حرارة مرتفعة — فضّل الصباح الباكر أو المساء للرعي، ووفّر الظل والماء.'
      : 'Forte chaleur : privilégiez pâturage tôt le matin ou en soirée, ombre et eau à disposition.';
  }
  if (cold) {
    return isAr
      ? 'برودة — قلّل مدة التعرّض في المرعى وراقب الإجهاد الحراري.'
      : 'Fraîcheur marquée : limitez le temps de pâturage et surveillez le confort thermique.';
  }
  if (code <= 3) {
    return isAr
      ? 'ظروف جيدة بشكل عام للرعي حسب حالة المراعي.'
      : 'Conditions globalement favorables au pâturage selon l’état des parcelles.';
  }
  return isAr
    ? 'راقب الطقس والمراعي قبل تحريك القطيع.'
    : 'Surveillez l’évolution du temps et l’état des parcelles avant la sortie.';
}

/** Icône Ionicons pour la carte météo (approximation selon code WMO). */
export function weatherCodeToIonName(code) {
  const c = code ?? 0;
  if (c === 0) return 'sunny-outline';
  if (c === 1) return 'partly-sunny-outline';
  if (c === 2) return 'cloud-outline';
  if (c === 3) return 'cloud';
  if (c >= 45 && c <= 48) return 'water-outline';
  if (c >= 51 && c <= 57) return 'rainy-outline';
  if (c >= 61 && c <= 67) return 'rainy-outline';
  if (c >= 71 && c <= 77) return 'snow-outline';
  if (c >= 80 && c <= 82) return 'rainy-outline';
  if (c >= 95) return 'thunderstorm-outline';
  return 'partly-cloudy-outline';
}

function formatLocalTime(isoTime) {
  if (!isoTime) return '';
  const d = new Date(isoTime);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
}

/**
 * @param {{ latitude: number, longitude: number }} coords
 * @returns {Promise<object>}
 */
export async function fetchOpenMeteoCurrent(coords) {
  const { latitude, longitude } = coords;
  const params = new URLSearchParams({
    latitude: String(latitude),
    longitude: String(longitude),
    current: [
      'temperature_2m',
      'apparent_temperature',
      'relative_humidity_2m',
      'weather_code',
      'wind_speed_10m',
      'wind_direction_10m',
    ].join(','),
    timezone: 'auto',
    wind_speed_unit: 'kmh',
  });

  const url = `${OPEN_METEO_BASE}?${params.toString()}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Open-Meteo HTTP ${response.status}`);
  }

  const data = await response.json();
  const cur = data.current;
  if (!cur) {
    throw new Error('Open-Meteo: pas de données current');
  }

  const code = cur.weather_code ?? 0;
  const summaryFr = wmoToSummary(code, 'fr');
  const summaryAr = wmoToSummary(code, 'ar');
  const windDeg = cur.wind_direction_10m ?? 0;
  const windDirFr = degreesToWindDir8(windDeg, 'fr');
  const windDirAr = degreesToWindDir8(windDeg, 'ar');

  const temp = Math.round(cur.temperature_2m ?? 0);
  const feels = Math.round(cur.apparent_temperature ?? temp);
  const humidity = Math.round(cur.relative_humidity_2m ?? 0);
  const windKmh = Math.round(cur.wind_speed_10m ?? 0);

  const pastureFr = pastureHintFromConditions(code, cur.temperature_2m ?? temp, 'fr');
  const pastureAr = pastureHintFromConditions(code, cur.temperature_2m ?? temp, 'ar');

  return {
    temp,
    feelsLike: feels,
    humidity,
    windKmh,
    windDirFr,
    windDirAr,
    summaryFr: summaryFr.text,
    summaryAr: summaryAr.text,
    icon: summaryFr.icon,
    pastureFr,
    pastureAr,
    updated: formatLocalTime(cur.time),
    rawTime: cur.time,
    weatherCode: code,
    latitude,
    longitude,
  };
}
