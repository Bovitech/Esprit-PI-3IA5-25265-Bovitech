import { Platform } from 'react-native';

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const r = reader.result;
      if (typeof r !== 'string') {
        reject(new Error('Lecture fichier audio impossible'));
        return;
      }
      const comma = r.indexOf(',');
      resolve(comma >= 0 ? r.slice(comma + 1) : r);
    };
    reader.onerror = () => reject(reader.error || new Error('FileReader'));
    reader.readAsDataURL(blob);
  });
}

/** Web: fetch blob; natif: expo-file-system/legacy (SDK 54). */
export async function recordingUriToBase64(uri) {
  if (Platform.OS === 'web') {
    const res = await fetch(uri);
    if (!res.ok) {
      throw new Error(`Impossible de lire l’audio (HTTP ${res.status})`);
    }
    const blob = await res.blob();
    return blobToBase64(blob);
  }
  const { readAsStringAsync } = await import('expo-file-system/legacy');
  return readAsStringAsync(uri, { encoding: 'base64' });
}
