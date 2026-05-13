/**
 * Implémentation web : pas d'expo-file-system/legacy (évite écran blanc au chargement).
 * Metro résout `chatbotApi.web.js` à la place de `chatbotApi.js` sur web.
 */
import { CHATBOT_API_BASE_URL } from '../config/api';

function buildUrl(path) {
  const base = CHATBOT_API_BASE_URL.replace(/\/+$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${p}`;
}

async function readStreamAsText(response) {
  if (response.body && typeof response.body.getReader === 'function') {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let content = '';
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      content += decoder.decode(value, { stream: true });
    }
    return content;
  }
  return response.text();
}

export async function sendChatMessage({
  message,
  sessionId,
  lang,
  lat = null,
  lon = null,
  onStreamChunk,
}) {
  const response = await fetch(buildUrl('/chatbot/'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      lang,
      lat,
      lon,
    }),
  });

  const contentType = response.headers.get('content-type') || '';

  if (!response.ok) {
    const errText = await response.text();
    try {
      const j = JSON.parse(errText);
      if (j.error) return { error: j.error };
    } catch {
      /* ignore */
    }
    throw new Error(errText || `Chat request failed (${response.status})`);
  }

  if (contentType.includes('application/json')) {
    return response.json();
  }

  if (typeof onStreamChunk === 'function' && response.body?.getReader) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let content = '';
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      content += chunk;
      onStreamChunk(chunk, content);
    }
    return { type: 'text', content };
  }

  const content = await readStreamAsText(response);
  if (typeof onStreamChunk === 'function' && content) {
    onStreamChunk(content, content);
  }
  return { type: 'text', content };
}

function extAndMimeFromHint(uriPath, mimeHint) {
  const lower = uriPath.toLowerCase();
  if (mimeHint) {
    const m = String(mimeHint).toLowerCase();
    if (m.includes('png')) return { ext: 'png', mime: 'image/png' };
    if (m.includes('webp')) return { ext: 'webp', mime: 'image/webp' };
    if (m.includes('gif')) return { ext: 'gif', mime: 'image/gif' };
    if (m.includes('jpeg') || m.includes('jpg')) return { ext: 'jpg', mime: 'image/jpeg' };
  }
  const ext = lower.endsWith('.png')
    ? 'png'
    : lower.endsWith('.webp')
      ? 'webp'
      : lower.endsWith('.gif')
        ? 'gif'
        : 'jpg';
  const mime =
    ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : ext === 'gif' ? 'image/gif' : 'image/jpeg';
  return { ext, mime };
}

export async function classifySkinImage({ imageUri, lang, message, mimeType }) {
  const { ext } = extAndMimeFromHint(imageUri.split('?')[0], mimeType);
  const formData = new FormData();
  formData.append('lang', String(lang));
  if (message) formData.append('message', String(message));

  const imgRes = await fetch(imageUri);
  const blob = await imgRes.blob();
  formData.append('image', blob, `bovitech-skin-${Date.now()}.${ext}`);

  const response = await fetch(buildUrl('/chatbot/skin/'), {
    method: 'POST',
    body: formData,
  });

  const text = await response.text();
  if (!response.ok) {
    try {
      const j = JSON.parse(text);
      if (j.error?.message) throw new Error(j.error.message);
    } catch (e) {
      if (e.message && e.message !== 'Unexpected token') throw e;
    }
    throw new Error(text || `Skin request failed (${response.status})`);
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new Error('Invalid skin classifier response.');
  }
}

function audioMultipartPart(uri) {
  const pathPart = uri.split('?')[0];
  const base = pathPart.split('/').pop() || 'audio.m4a';
  const dot = base.lastIndexOf('.');
  const ext = dot >= 0 ? base.slice(dot + 1).toLowerCase() : 'm4a';
  const safeExt = ['m4a', 'caf', 'wav', 'mp3', 'mp4', 'aac', 'webm'].includes(ext) ? ext : 'm4a';
  const mime = {
    m4a: 'audio/m4a',
    caf: 'audio/x-caf',
    wav: 'audio/wav',
    mp3: 'audio/mpeg',
    mp4: 'audio/mp4',
    aac: 'audio/aac',
    webm: 'audio/webm',
  }[safeExt] || 'audio/m4a';
  return { name: `audio.${safeExt}`, type: mime };
}

export async function transcribeAudio({ uri, lang }) {
  const { name } = audioMultipartPart(uri);
  const audioRes = await fetch(uri);
  const blob = await audioRes.blob();
  const formData = new FormData();
  formData.append('lang', lang);
  formData.append('audio', blob, name);

  const response = await fetch(buildUrl('/chatbot/stt/'), {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error?.message || `STT failed (${response.status})`);
  }
  return data;
}

export async function synthesizeToFile({ text, lang }) {
  const response = await fetch(buildUrl('/chatbot/tts/'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang }),
  });

  if (!response.ok) {
    const errText = await response.text();
    try {
      const j = JSON.parse(errText);
      throw new Error(j?.error?.message || 'TTS failed');
    } catch (e) {
      if (e.message && !errText.includes('{')) throw e;
      throw new Error(errText || 'TTS failed');
    }
  }

  const ct = (response.headers.get('content-type') || '').toLowerCase();
  const mime =
    ct.includes('mpeg') || ct.includes('mp3') || ct.includes('audio/mp3') ? 'audio/mpeg' : 'audio/wav';

  const ab = await response.arrayBuffer();
  const blob = new Blob([ab], { type: mime });
  return URL.createObjectURL(blob);
}
