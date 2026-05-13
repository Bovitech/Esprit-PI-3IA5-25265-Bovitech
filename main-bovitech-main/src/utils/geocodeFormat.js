/**
 * Formate le résultat d'expo-location reverseGeocodeAsync pour l'UI (libellés texte uniquement).
 * @param {import('expo-location').LocationGeocodedAddress | null | undefined} geo
 * @returns {{ header: string | null, card: string | null }}
 */
export function linesFromGeocode(geo) {
  if (!geo) return { header: null, card: null };

  const locality = geo.city || geo.district || geo.subregion || null;
  const region = geo.region || geo.subregion || null;
  const streetPart = [geo.streetNumber, geo.street].filter(Boolean).join(' ').trim() || null;
  const formatted = geo.formattedAddress?.trim() || null;

  let card = null;
  if (streetPart && locality) {
    card = `${streetPart}, ${locality}`;
  } else if (locality && region && locality !== region) {
    card = `${locality} · ${region}`;
  } else if (locality) {
    card = locality;
  } else if (geo.name) {
    card = geo.name;
  } else if (region) {
    card = region;
  } else if (geo.country) {
    card = geo.country;
  }

  let header = null;
  if (locality && region && locality !== region) {
    header = `${locality} · ${region}`;
  } else {
    header = locality || region || geo.country || geo.name || null;
  }

  if (formatted) {
    if (!card) card = formatted;
    if (!header) {
      const parts = formatted.split(',').map((s) => s.trim()).filter(Boolean);
      header = parts.length >= 2 ? parts.slice(-2).join(' · ') : formatted;
    }
  }

  return { header, card };
}

/**
 * Secours : nom de lieu lisible (ville / région / pays) sans coordonnées.
 * API client BigDataCloud (gratuite, sans clé).
 * @param {{ latitude: number, longitude: number }} coords
 * @param {'fr'|'ar'} lang
 * @returns {Promise<{ header: string | null, card: string | null } | null>}
 */
export async function fetchNamedPlaceLinesFromCoords(coords, lang) {
  const lat = Number(coords.latitude);
  const lon = Number(coords.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  const localityLanguage = lang === 'ar' ? 'ar' : 'fr';
  const q = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lon),
    localityLanguage,
  });

  try {
    const res = await fetch(
      `https://api.bigdatacloud.net/data/reverse-geocode-client?${q.toString()}`
    );
    if (!res.ok) return null;
    const j = await res.json();
    const locality = (j.locality || '').trim();
    const city = (j.city || '').trim();
    const prov = (j.principalSubdivision || '').trim();
    const country = (j.countryName || '').trim();

    const locCity =
      locality && city && locality !== city
        ? `${locality}, ${city}`
        : city || locality || '';

    let card = null;
    if (locCity && prov) card = `${locCity} · ${prov}`;
    else if (locCity) card = locCity;
    else if (prov && country) card = `${prov} · ${country}`;
    else if (country) card = country;
    else if (prov) card = prov;

    let header = null;
    if (city && prov && city !== prov) header = `${city} · ${prov}`;
    else if (locality && prov && locality !== prov) header = `${locality} · ${prov}`;
    else header = card;

    if (!header && !card) return null;
    return { header: header || card, card: card || header };
  } catch {
    return null;
  }
}
