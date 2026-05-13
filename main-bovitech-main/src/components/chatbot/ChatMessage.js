import React, { memo, useEffect, useRef } from 'react';
import {
  Animated,
  Image,
  Linking,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AIBubble from './AIBubble';

const COLORS = {
  primary: '#2E412D',
  cream: '#F6F6EC',
  discussion: '#E9E9E0',
  sage: '#AFB2A1',
  dark: '#3E3E32',
  black: '#010101',
  white: '#FFFFFF',
  border: '#DAD9CF',
};

function stripTrailingPunctuation(url) {
  return url.replace(/[.,);:]+$/g, '');
}

/** Lat,lng depuis une URL Google Maps (query= ou q=). */
function extractMapsCoords(url) {
  const u = stripTrailingPunctuation(url);
  const q = u.match(/[?&]query=([-0-9.]+),([-0-9.]+)/i);
  if (q) return { lat: q[1], lng: q[2] };
  const qq = u.match(/[?&]q=([-0-9.]+),([-0-9.]+)/i);
  if (qq) return { lat: qq[1], lng: qq[2] };
  return null;
}

function isGoogleMapsHttpUrl(url) {
  return /google\.com(\/.*)?\/maps|maps\.google\./i.test(stripTrailingPunctuation(url));
}

/**
 * Ouvre l’app Cartes / Google Maps plutôt que le navigateur lorsque c’est possible.
 */
async function openLinkPreferNativeMaps(rawUrl) {
  const clean = stripTrailingPunctuation(rawUrl);
  const coords = extractMapsCoords(clean);

  if (coords && isGoogleMapsHttpUrl(clean)) {
    const { lat, lng } = coords;
    const label = encodeURIComponent('Vétérinaire');

    if (Platform.OS === 'android') {
      const geo = `geo:${lat},${lng}?q=${lat},${lng}(${label})`;
      try {
        await Linking.openURL(geo);
        return;
      } catch {
        /* fallback https */
      }
    }

    if (Platform.OS === 'ios') {
      const appleMaps = `maps://maps.apple.com/?ll=${lat},${lng}&q=${label}`;
      const googleMaps = `comgooglemaps://?q=${lat},${lng}`;
      try {
        const canGoogle = await Linking.canOpenURL(googleMaps);
        if (canGoogle) {
          await Linking.openURL(googleMaps);
          return;
        }
      } catch {
        /* ignore */
      }
      try {
        await Linking.openURL(appleMaps);
        return;
      } catch {
        /* fallback https */
      }
    }
  }

  try {
    await Linking.openURL(clean);
  } catch {
    // dernier recours : URL brute
    Linking.openURL(rawUrl);
  }
}

function parseTextWithUrls(text) {
  if (!text) return [];
  const re = /https?:\/\/[^\s]+/gi;
  const parts = [];
  let last = 0;
  let m = re.exec(text);
  while (m !== null) {
    if (m.index > last) {
      parts.push({ type: 'text', value: text.slice(last, m.index) });
    }
    parts.push({ type: 'url', value: m[0] });
    last = m.index + m[0].length;
    m = re.exec(text);
  }
  if (last < text.length) {
    parts.push({ type: 'text', value: text.slice(last) });
  }
  return parts;
}

const MessageBodyText = memo(function MessageBodyText({
  children,
  baseStyle,
  linkStyle,
  streaming,
}) {
  const text = typeof children === 'string' ? children : '';
  const segments = parseTextWithUrls(text);

  if (segments.length === 0) {
    return (
      <Text style={baseStyle}>
        {streaming ? '▋' : ''}
      </Text>
    );
  }

  return (
    <Text style={baseStyle}>
      {segments.map((seg, i) => {
        if (seg.type === 'url') {
          return (
            <Text
              key={`u-${i}`}
              style={[baseStyle, linkStyle]}
              onPress={() => openLinkPreferNativeMaps(seg.value)}
            >
              {seg.value}
            </Text>
          );
        }
        return seg.value;
      })}
      {streaming ? '▋' : ''}
    </Text>
  );
});

const TypingDots = memo(function TypingDots() {
  const one = useRef(new Animated.Value(0.25)).current;
  const two = useRef(new Animated.Value(0.25)).current;
  const three = useRef(new Animated.Value(0.25)).current;

  useEffect(() => {
    const makeLoop = (value, delay) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(value, {
            toValue: 1,
            duration: 320,
            useNativeDriver: true,
          }),
          Animated.timing(value, {
            toValue: 0.25,
            duration: 320,
            useNativeDriver: true,
          }),
        ])
      );

    const a = makeLoop(one, 0);
    const b = makeLoop(two, 120);
    const c = makeLoop(three, 240);

    a.start();
    b.start();
    c.start();

    return () => {
      a.stop();
      b.stop();
      c.stop();
    };
  }, [one, two, three]);

  return (
    <View style={styles.dots}>
      {[one, two, three].map((dot, index) => (
        <Animated.View key={index} style={[styles.dot, { opacity: dot }]} />
      ))}
    </View>
  );
});

const ChatMessage = memo(function ChatMessage({
  message,
  onSpeak,
  onStopSpeaking,
  speakingMessageId,
}) {
  const isUser = message.role === 'user';
  const isTyping = message.type === 'typing';
  const isSpeaking = speakingMessageId === message.id;

  if (!isTyping && !message.text && !message.imageUri) {
    return null;
  }

  return (
    <View style={[styles.row, isUser && styles.rowUser]}>
      {!isUser && (
        <View style={styles.avatarWrap}>
          <AIBubble size={42} compact active={isTyping || isSpeaking || message.streaming} />
        </View>
      )}

      <View style={[styles.bubble, isUser ? styles.userBubble : styles.botBubble]}>
        {isTyping ? (
          <TypingDots />
        ) : (
          <>
            {!!message.imageUri && (
              <Image source={{ uri: message.imageUri }} style={styles.imagePreview} />
            )}

            {!!message.text && (
              <MessageBodyText
                baseStyle={[styles.text, isUser ? styles.userText : styles.botText]}
                linkStyle={isUser ? styles.userLink : styles.botLink}
                streaming={!!message.streaming}
              >
                {message.text}
              </MessageBodyText>
            )}

            <View style={styles.footer}>
              {!isUser && !!message.text && !message.streaming && (
                <TouchableOpacity
                  style={styles.speakBtn}
                  onPress={() => {
                    if (isSpeaking) {
                      onStopSpeaking?.();
                    } else {
                      onSpeak?.(message);
                    }
                  }}
                >
                  <Ionicons
                    name={isSpeaking ? 'stop-circle' : 'volume-medium-outline'}
                    size={15}
                    color={COLORS.primary}
                  />

                  <Text style={styles.speakText}>{isSpeaking ? 'Stop' : 'Écouter'}</Text>
                </TouchableOpacity>
              )}

              {!!message.time && (
                <Text style={[styles.time, isUser ? styles.userTime : styles.botTime]}>
                  {message.time}
                </Text>
              )}
            </View>
          </>
        )}
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
    marginBottom: 12,
  },

  rowUser: {
    justifyContent: 'flex-end',
  },

  avatarWrap: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },

  bubble: {
    maxWidth: '78%',
    borderRadius: 22,
    paddingHorizontal: 14,
    paddingVertical: 11,
  },

  botBubble: {
    backgroundColor: COLORS.white,
    borderBottomLeftRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  userBubble: {
    backgroundColor: COLORS.primary,
    borderBottomRightRadius: 8,
  },

  text: {
    fontFamily: 'Inter_500Medium',
    fontSize: 14,
    lineHeight: 20,
  },

  botText: {
    color: COLORS.dark,
  },

  botLink: {
    color: '#1d4ed8',
    textDecorationLine: 'underline',
  },

  userText: {
    color: COLORS.white,
  },

  userLink: {
    color: '#bfdbfe',
    textDecorationLine: 'underline',
  },

  imagePreview: {
    width: 190,
    height: 150,
    borderRadius: 18,
    marginBottom: 8,
    backgroundColor: COLORS.discussion,
  },

  footer: {
    marginTop: 7,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  speakBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
    backgroundColor: COLORS.discussion,
  },

  speakText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 10,
    color: COLORS.primary,
  },

  time: {
    fontFamily: 'Inter_500Medium',
    fontSize: 10,
    alignSelf: 'flex-end',
  },

  botTime: {
    color: COLORS.sage,
  },

  userTime: {
    color: 'rgba(255,255,255,0.72)',
  },

  dots: {
    flexDirection: 'row',
    gap: 5,
    paddingVertical: 4,
    paddingHorizontal: 4,
  },

  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: COLORS.primary,
  },
});

export default ChatMessage;