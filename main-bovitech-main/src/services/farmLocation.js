import * as Location from 'expo-location';

import { linesFromGeocode, fetchNamedPlaceLinesFromCoords } from '../utils/geocodeFormat';
import { DEFAULT_FARM_COORDS } from './openMeteo';

let cached = null;
let inflight = null;

const FALLBACK_LABEL_FR = 'Position par défaut (Tunisie)';
const FALLBACK_LABEL_AR = 'موقع افتراضي (تونس)';

/**
 * @typedef {{
 *   coords: { latitude: number, longitude: number },
 *   label: string,
 *   usedFallbackCoords: boolean,
 *   permissionDenied: boolean,
 * }} FarmLocationResult
 */

/** Pour tests ou invalidation après changement de permission. */
export function clearFarmLocationCache() {
  cached = null;
  inflight = null;
}

/**
 * Une résolution partagée (cache) : permission → GPS → géocodage inverse.
 * @param {'fr'|'ar'} lang
 * @returns {Promise<FarmLocationResult>}
 */
export async function resolveFarmLocation(lang = 'fr') {
  if (cached) return cached;
  if (inflight) return inflight;

  const fallbackLabel = lang === 'ar' ? FALLBACK_LABEL_AR : FALLBACK_LABEL_FR;

  inflight = (async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        cached = {
          coords: { ...DEFAULT_FARM_COORDS },
          label: fallbackLabel,
          usedFallbackCoords: true,
          permissionDenied: true,
        };
        return cached;
      }

      const pos = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });
      const coords = {
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
      };

      let label = null;
      try {
        const geos = await Location.reverseGeocodeAsync(coords);
        const lines = linesFromGeocode(geos?.[0]);
        label = lines.card || lines.header;
      } catch {
        /* BigDataCloud en secours */
      }
      if (!label) {
        const named = await fetchNamedPlaceLinesFromCoords(coords, lang);
        if (named) label = named.card || named.header;
      }
      if (!label) {
        label =
          lang === 'ar'
            ? `${coords.latitude.toFixed(4)}، ${coords.longitude.toFixed(4)}`
            : `${coords.latitude.toFixed(4)}, ${coords.longitude.toFixed(4)}`;
      }

      cached = {
        coords,
        label,
        usedFallbackCoords: false,
        permissionDenied: false,
      };
      return cached;
    } catch {
      cached = {
        coords: { ...DEFAULT_FARM_COORDS },
        label: fallbackLabel,
        usedFallbackCoords: true,
        permissionDenied: false,
      };
      return cached;
    } finally {
      inflight = null;
    }
  })();

  return inflight;
}
