import { useEffect, useState } from 'react';

import { DEFAULT_FARM_COORDS } from '../services/openMeteo';
import { resolveFarmLocation } from '../services/farmLocation';

/**
 * Position ferme / utilisateur pour météo et libellés UI.
 * @param {'fr'|'ar'} lang
 */
export function useFarmLocation(lang = 'fr') {
  const [state, setState] = useState(() => ({
    loading: true,
    coords: DEFAULT_FARM_COORDS,
    label: '',
    usedFallbackCoords: true,
    permissionDenied: false,
  }));

  useEffect(() => {
    let cancelled = false;
    resolveFarmLocation(lang).then((r) => {
      if (cancelled) return;
      setState({
        loading: false,
        coords: r.coords,
        label: r.label,
        usedFallbackCoords: r.usedFallbackCoords,
        permissionDenied: r.permissionDenied,
      });
    });
    return () => {
      cancelled = true;
    };
  }, [lang]);

  return state;
}
