import { Platform, Alert } from 'react-native';

export function getAuthErrorMessage(err, fallback = 'Request failed.') {
  if (err?.response?.data?.error) {
    return err.response.data.error;
  }
  if (err?.message === 'Network Error' || err?.code === 'ERR_NETWORK') {
    return 'Cannot reach PI_Backend. Start it: cd PI_Backend && python manage.py runserver (port 8000).';
  }
  if (err?.message) {
    return err.message;
  }
  return fallback;
}

export function showFormError(title, message, setInlineError) {
  if (setInlineError) {
    setInlineError(message);
  }
  if (Platform.OS !== 'web') {
    Alert.alert(title, message);
  }
}
