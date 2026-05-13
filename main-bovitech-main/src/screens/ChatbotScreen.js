import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as Speech from 'expo-speech';

import AIBubble from '../components/chatbot/AIBubble';
import ChatMessage from '../components/chatbot/ChatMessage';
import QuickAction from '../components/chatbot/QuickAction';
import { CHATBOT_API_BASE_URL } from '../config/api';
import { classifySkinImage } from '../services/chatbotApi';
import { resolveFarmLocation } from '../services/farmLocation';

const CHATBOT_ORIGIN = CHATBOT_API_BASE_URL.replace(/\/+$/, '');
const CHAT_ENDPOINT = `${CHATBOT_ORIGIN}/chatbot/`;

const COLORS = {
  primary: '#2E412D',
  earth: '#7C6144',
  cream: '#F6F6EC',
  discussion: '#E9E9E0',
  sage: '#AFB2A1',
  dark: '#3E3E32',
  black: '#010101',
  white: '#FFFFFF',
  border: '#DAD9CF',
  success: '#4F8F63',
  warning: '#D9962F',
};

function getTime() {
  const date = new Date();

  return `${String(date.getHours()).padStart(2, '0')}:${String(
    date.getMinutes()
  ).padStart(2, '0')}`;
}

function makeId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getBackendErrorMessage() {
  return "Impossible de joindre le backend RAG pour le moment. Vérifie que Django est lancé et que l’endpoint /chatbot/ répond.";
}

function extractSSEText(chunk) {
  return String(chunk || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      if (line.startsWith('data:')) {
        return line.replace(/^data:\s?/, '');
      }

      return line;
    })
    .filter((line) => line !== '[DONE]')
    .map((line) => {
      try {
        const parsed = JSON.parse(line);

        return (
          parsed.content ||
          parsed.reply ||
          parsed.response ||
          parsed.answer ||
          parsed.message ||
          parsed.text ||
          ''
        );
      } catch {
        return line;
      }
    })
    .join('');
}

function formatAgentReply(payload) {
  const agent = payload?.agent;
  const data = payload?.data || payload;

  if (!data || typeof data !== 'object') return '';

  if (data.decision || data.reason || data.tip) {
    const decisionText =
      data.decision === 'out'
        ? 'Oui, les vaches peuvent sortir.'
        : data.decision === 'in'
          ? 'Non, il vaut mieux garder les vaches à l’intérieur.'
          : '';

    return [
      decisionText,
      data.reason ? `Raison : ${data.reason}` : '',
      typeof data.temp !== 'undefined' ? `Température : ${data.temp}°C.` : '',
      typeof data.rain !== 'undefined' ? `Pluie : ${data.rain} mm.` : '',
      typeof data.wind !== 'undefined' ? `Vent : ${data.wind} km/h.` : '',
      data.tip ? `Conseil : ${data.tip}` : '',
    ]
      .filter(Boolean)
      .join('\n');
  }

  // Avant le bloc nutrition : `data.warning` existe aussi pour vet (>10 km), sinon mauvaise réponse.
  if (agent === 'vet') {
    // Contrat Django : { found, best: { name, distance_km, phone, map_url }, others[], warning }
    if (data.found && data.best) {
      const b = data.best;
      const lines = [
        `Vétérinaire le plus proche : ${b.name || 'Cabinet vétérinaire'} (${b.distance_km} km).`,
        b.phone ? `Téléphone : ${b.phone}.` : '',
        b.map_url ? `Carte : ${b.map_url}` : '',
      ];
      if (Array.isArray(data.others) && data.others.length > 0) {
        lines.push(
          `Autres cabinets : ${data.others.map((o) => `${o.name} (${o.distance_km} km)`).join(' · ')}.`
        );
      }
      if (data.warning) lines.push(data.warning);
      return lines.filter(Boolean).join('\n');
    }

    if (data.found === false) {
      return data.warning || 'Aucun vétérinaire proche trouvé pour le moment.';
    }

    if (Array.isArray(data.results) && data.results.length > 0) {
      const first = data.results[0];

      return `J’ai trouvé un vétérinaire proche : ${
        first.name || 'Clinique vétérinaire'
      }. ${first.address ? `Adresse : ${first.address}. ` : ''}${
        first.phone ? `Téléphone : ${first.phone}.` : ''
      }`;
    }

    if (Array.isArray(data.vets) && data.vets.length > 0) {
      const first = data.vets[0];

      return `J’ai trouvé un vétérinaire proche : ${
        first.name || 'Clinique vétérinaire'
      }. ${first.address ? `Adresse : ${first.address}. ` : ''}${
        first.phone ? `Téléphone : ${first.phone}.` : ''
      }`;
    }

    return data.message || 'Aucun vétérinaire proche trouvé pour le moment.';
  }

  if (data.main_feed || data.supplement || data.water || data.warning) {
    return [
      data.season ? `Plan nutrition — saison : ${data.season}.` : 'Plan nutrition recommandé.',
      data.main_feed ? `Aliment principal : ${data.main_feed}` : '',
      data.supplement ? `Supplément : ${data.supplement}` : '',
      data.water ? `Eau : ${data.water}` : '',
      data.warning ? `Attention : ${data.warning}` : '',
      data.tip ? `Conseil : ${data.tip}` : '',
    ]
      .filter(Boolean)
      .join('\n');
  }

  if (agent === 'meteo') {
    return (
      data.summary ||
      data.message ||
      data.description ||
      data.forecast ||
      ''
    );
  }

  if (agent === 'feed') {
    return (
      data.recommendation ||
      data.message ||
      data.summary ||
      data.advice ||
      ''
    );
  }

  if (agent === 'skin') {
    const confidence =
      typeof data.confidence === 'number'
        ? `${Math.round(data.confidence * 100)}%`
        : '?';

    return `Analyse photo : ${
      data.predicted_class || data.class || 'résultat inconnu'
    } avec une confiance de ${confidence}. ${data.description || data.message || ''}`;
  }

  return (
    data.message ||
    data.summary ||
    data.description ||
    data.recommendation ||
    data.advice ||
    ''
  );
}

function extractReplyFromJson(payload) {
  if (!payload) return '';

  if (typeof payload === 'string') return payload;

  if (payload.error?.message) {
    return payload.error.message;
  }

  if (payload.type === 'text') {
    return payload.content || payload.text || payload.message || '';
  }

  if (payload.type === 'agent') {
    return formatAgentReply(payload);
  }

  const directReply =
    payload.content ||
    payload.reply ||
    payload.response ||
    payload.answer ||
    payload.message ||
    payload.text ||
    payload.data?.content ||
    payload.data?.reply ||
    payload.data?.response ||
    payload.data?.answer;

  if (directReply) return directReply;

  const formatted = formatAgentReply(payload);
  if (formatted) return formatted;

  return "J’ai reçu une réponse du backend, mais son format n’est pas encore reconnu par l’interface.";
}

const initialMessages = [
  {
    id: 'welcome',
    role: 'assistant',
    text:
      "Bonjour, je suis votre assistant BoviTech. Je peux vous aider pour la santé du troupeau, la météo, l’alimentation, la production de lait et l’analyse photo.",
    time: getTime(),
  },
];

export default function ChatbotScreen() {
  const listRef = useRef(null);
  const streamTimerRef = useRef(null);
  const bubbleIntro = useRef(new Animated.Value(0)).current;

  const [input, setInput] = useState('');
  const [messages, setMessages] = useState(initialMessages);
  const [isTyping, setIsTyping] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState(null);
  const [backendStatus, setBackendStatus] = useState('Connecté');

  useEffect(() => {
    Animated.spring(bubbleIntro, {
      toValue: 1,
      friction: 8,
      tension: 70,
      useNativeDriver: true,
    }).start();

    return () => {
      if (streamTimerRef.current) {
        clearInterval(streamTimerRef.current);
      }

      Speech.stop();
    };
  }, [bubbleIntro]);

  const quickActions = useMemo(
    () => [
      {
        icon: 'heart-pulse',
        lib: 'mc',
        label: 'Santé vache',
        prompt: 'Analyse la santé de Rosette avec température élevée.',
      },
      {
        icon: 'cloud-outline',
        lib: 'ion',
        label: 'Météo ferme',
        prompt: 'Donne-moi un conseil météo pour la ferme aujourd’hui.',
      },
      {
        icon: 'camera-outline',
        lib: 'ion',
        label: 'Analyse photo',
        prompt: 'Analyse cette photo et dis-moi s’il y a un signe inquiétant.',
      },
      {
        icon: 'nutrition-outline',
        lib: 'ion',
        label: 'Alimentation',
        prompt: 'Donne-moi un conseil alimentation pour améliorer la production de lait.',
      },
    ],
    []
  );

  const scrollToEnd = useCallback(() => {
    setTimeout(() => {
      listRef.current?.scrollToEnd?.({ animated: true });
    }, 80);
  }, []);

  const addTypingMessage = useCallback(() => {
    setIsTyping(true);

    setMessages((prev) => {
      const withoutTyping = prev.filter((message) => message.id !== 'typing');

      return withoutTyping.concat({
        id: 'typing',
        role: 'assistant',
        type: 'typing',
        time: getTime(),
      });
    });

    scrollToEnd();
  }, [scrollToEnd]);

  const createAssistantMessage = useCallback(
    (replyId) => {
      setMessages((prev) =>
        prev.filter((message) => message.id !== 'typing').concat({
          id: replyId,
          role: 'assistant',
          text: '',
          time: getTime(),
          streaming: true,
        })
      );

      setIsTyping(false);
      scrollToEnd();
    },
    [scrollToEnd]
  );

  const updateAssistantMessage = useCallback(
    (replyId, nextText, done = false) => {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === replyId
            ? {
                ...message,
                text: nextText,
                streaming: !done,
              }
            : message
        )
      );

      scrollToEnd();
    },
    [scrollToEnd]
  );

  const streamReply = useCallback(
    (replyText) => {
      const replyId = makeId('assistant');

      addTypingMessage();

      setTimeout(() => {
        createAssistantMessage(replyId);

        let index = 0;
        let currentText = '';

        if (streamTimerRef.current) {
          clearInterval(streamTimerRef.current);
        }

        streamTimerRef.current = setInterval(() => {
          currentText += replyText.slice(index, index + 3);
          index += 3;

          const done = index >= replyText.length;

          updateAssistantMessage(replyId, currentText, done);

          if (done && streamTimerRef.current) {
            clearInterval(streamTimerRef.current);
            streamTimerRef.current = null;
          }
        }, 18);
      }, 260);
    },
    [addTypingMessage, createAssistantMessage, updateAssistantMessage]
  );

  const askBackend = useCallback(
    async ({ text, hasImage, imageUri, imageMimeType }) => {
      const replyId = makeId('assistant');

      addTypingMessage();

      try {
        if (hasImage && imageUri) {
          const payload = await classifySkinImage({
            imageUri,
            lang: 'fr',
            message: text,
            mimeType: imageMimeType,
          });

          setBackendStatus('Connecté');
          createAssistantMessage(replyId);

          const replyText = extractReplyFromJson(payload);

          updateAssistantMessage(
            replyId,
            replyText || "Réponse analyse peau vide — vérifiez le format de l'image.",
            true
          );

          return;
        }

        const loc = await resolveFarmLocation('fr');
        const { latitude: lat, longitude: lon } = loc.coords;

        const response = await fetch(CHAT_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream, application/json, text/plain',
          },
          body: JSON.stringify({
            message: text,
            session_id: 'mobile-front',
            lang: 'fr',
            lat,
            lon,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        setBackendStatus('Connecté');
        createAssistantMessage(replyId);

        const contentType = response.headers.get('content-type') || '';

        if (contentType.includes('application/json')) {
          const payload = await response.json();
          const replyText = extractReplyFromJson(payload);

          updateAssistantMessage(
            replyId,
            replyText || getBackendErrorMessage(),
            true
          );

          return;
        }

        if (response.body?.getReader) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');

          let done = false;
          let fullText = '';

          while (!done) {
            const result = await reader.read();
            done = result.done;

            const chunk = decoder.decode(result.value || new Uint8Array(), {
              stream: !done,
            });

            const nextText = extractSSEText(chunk);

            if (nextText) {
              fullText += nextText;
              updateAssistantMessage(replyId, fullText, false);
            }
          }

          updateAssistantMessage(
            replyId,
            fullText || getBackendErrorMessage(),
            true
          );

          return;
        }

        const plainText = await response.text();

        updateAssistantMessage(
          replyId,
          plainText || getBackendErrorMessage(),
          true
        );
      } catch (error) {
        const raw = error?.message ? String(error.message) : String(error);
        const looksUnreachable =
          /network request failed|failed to fetch|network error|offline|fetch/i.test(
            raw
          ) || raw.includes('ECONNREFUSED');
        const userText = looksUnreachable ? getBackendErrorMessage() : raw;

        setBackendStatus('Backend non disponible');
        setMessages((prev) => prev.filter((message) => message.id !== 'typing'));
        setIsTyping(false);

        streamReply(userText);
      }
    },
    [addTypingMessage, createAssistantMessage, streamReply, updateAssistantMessage]
  );

  const sendMessage = useCallback(
    async (forcedText) => {
      const text = (forcedText ?? input).trim();
      const hasImage = !!selectedImage;

      if ((!text && !hasImage) || isTyping) return;

      const finalText = text || 'Analyse cette image.';

      const userMessage = {
        id: makeId('user'),
        role: 'user',
        text: finalText,
        imageUri: selectedImage?.uri,
        time: getTime(),
      };

      setMessages((prev) => prev.concat(userMessage));
      setInput('');
      setSelectedImage(null);
      scrollToEnd();

      await askBackend({
        text: finalText,
        hasImage,
        imageUri: selectedImage?.uri,
        imageMimeType: selectedImage?.mimeType || selectedImage?.type,
      });
    },
    [askBackend, input, isTyping, scrollToEnd, selectedImage]
  );

  const pickImage = useCallback(async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      streamReply("Autorise l’accès aux photos pour envoyer une image au backend.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.78,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets?.[0]) {
      setSelectedImage(result.assets[0]);
    }
  }, [streamReply]);

  const handleSpeak = useCallback((message) => {
    Speech.stop();
    setSpeakingMessageId(message.id);

    Speech.speak(message.text, {
      language: 'fr-FR',
      pitch: 1,
      rate: 0.94,
      onDone: () => setSpeakingMessageId(null),
      onStopped: () => setSpeakingMessageId(null),
      onError: () => setSpeakingMessageId(null),
    });
  }, []);

  const handleStopSpeaking = useCallback(() => {
    Speech.stop();
    setSpeakingMessageId(null);
  }, []);

  const handleMicPress = useCallback(() => {
    setIsListening((prev) => !prev);

    if (!isListening) {
      setInput('Cette vache a faim ?');
    }
  }, [isListening]);

  const renderMessage = useCallback(
    ({ item }) => (
      <ChatMessage
        message={item}
        onSpeak={handleSpeak}
        onStopSpeaking={handleStopSpeaking}
        speakingMessageId={speakingMessageId}
      />
    ),
    [handleSpeak, handleStopSpeaking, speakingMessageId]
  );

  const bubbleScale = bubbleIntro.interpolate({
    inputRange: [0, 1],
    outputRange: [0.86, 1],
  });

  const bubbleOpacity = bubbleIntro.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        style={styles.screen}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 8 : 0}
      >
        <View style={styles.header}>
          <TouchableOpacity style={styles.headerBtn}>
            <Ionicons name="menu-outline" size={22} color={COLORS.primary} />
          </TouchableOpacity>

          <View style={styles.brandBlock}>
            <Text style={styles.assistantLabel}>ASSISTANT IA</Text>
            <Text style={styles.brand}>BoviTech</Text>
          </View>

          <TouchableOpacity style={styles.headerBtn}>
            <Ionicons name="settings-outline" size={20} color={COLORS.primary} />
          </TouchableOpacity>
        </View>

        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={renderMessage}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={styles.listContent}
          ListFooterComponent={<View style={{ height: 28 }} />}
          ListHeaderComponent={
            <View style={styles.hero}>
              <Text style={styles.heroTitle}>Comment puis-je vous aider aujourd’hui ?</Text>

              <Animated.View
                style={[
                  styles.bigBubbleWrap,
                  {
                    opacity: bubbleOpacity,
                    transform: [{ scale: bubbleScale }],
                  },
                ]}
              >
                <AIBubble size={170} active />
              </Animated.View>

              <View style={styles.quickGrid}>
                {quickActions.map((item) => (
                  <QuickAction
                    key={item.label}
                    icon={item.icon}
                    lib={item.lib}
                    label={item.label}
                    onPress={() => sendMessage(item.prompt)}
                  />
                ))}
              </View>

              <Text style={styles.discussionTitle}>Discussion</Text>
            </View>
          }
          onContentSizeChange={scrollToEnd}
        />

        {selectedImage && (
          <View style={styles.imagePreviewBar}>
            <Image source={{ uri: selectedImage.uri }} style={styles.selectedImage} />

            <View style={{ flex: 1 }}>
              <Text style={styles.imageTitle}>Image sélectionnée</Text>
              <Text style={styles.imageText}>Elle sera envoyée avec votre message.</Text>
            </View>

            <TouchableOpacity style={styles.removeImageBtn} onPress={() => setSelectedImage(null)}>
              <Ionicons name="close" size={18} color={COLORS.primary} />
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.inputOuter}>
          <View style={styles.inputBar}>
            <TouchableOpacity style={styles.inputIconBtn} onPress={pickImage}>
              <Ionicons name="image-outline" size={21} color={COLORS.primary} />
            </TouchableOpacity>

            <TextInput
              value={input}
              onChangeText={setInput}
              placeholder="Posez votre question..."
              placeholderTextColor={COLORS.sage}
              style={styles.input}
              multiline={false}
              returnKeyType="send"
              blurOnSubmit={false}
              onSubmitEditing={() => sendMessage()}
            />

            <TouchableOpacity
              style={[styles.micBtn, isListening && styles.micBtnActive]}
              onPress={handleMicPress}
            >
              <Ionicons name={isListening ? 'radio' : 'mic'} size={19} color={COLORS.white} />
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.sendBtn, isTyping && styles.sendBtnDisabled]}
              onPress={() => sendMessage()}
              disabled={isTyping}
              activeOpacity={0.86}
            >
              <Ionicons name="send" size={21} color={COLORS.white} />
            </TouchableOpacity>
          </View>

          <View style={styles.statusRow}>
            <View
              style={[
                styles.statusDot,
                backendStatus !== 'Connecté' && styles.statusDotError,
                isTyping && styles.statusDotBusy,
              ]}
            />

            <Text style={styles.statusText}>
              {isTyping ? 'Réponse en cours...' : `${backendStatus} · ${CHATBOT_ORIGIN}`}
            </Text>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.discussion,
  },

  screen: {
    flex: 1,
    backgroundColor: COLORS.discussion,
  },

  header: {
    height: 54,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  headerBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: COLORS.cream,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
  },

  brandBlock: {
    alignItems: 'center',
  },

  assistantLabel: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 10,
    letterSpacing: 1.4,
    color: COLORS.sage,
  },

  brand: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 21,
    color: COLORS.primary,
    letterSpacing: -0.4,
    marginTop: -2,
  },

  listContent: {
    paddingHorizontal: 18,
    paddingBottom: 135,
  },

  hero: {
    alignItems: 'center',
    paddingTop: 4,
  },

  heroTitle: {
    width: '90%',
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 24,
    lineHeight: 30,
    color: COLORS.primary,
    textAlign: 'center',
    letterSpacing: -0.6,
    marginTop: 4,
  },

  bigBubbleWrap: {
    marginTop: 18,
    marginBottom: 18,
  },

  quickGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 10,
    marginBottom: 18,
  },

  discussionTitle: {
    alignSelf: 'flex-start',
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 19,
    color: COLORS.black,
    marginBottom: 14,
  },

  imagePreviewBar: {
    position: 'absolute',
    left: 14,
    right: 14,
    bottom: Platform.OS === 'ios' ? 158 : 148,
    borderRadius: 20,
    backgroundColor: COLORS.cream,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    zIndex: 60,
    elevation: 60,
  },

  selectedImage: {
    width: 48,
    height: 48,
    borderRadius: 14,
    backgroundColor: COLORS.discussion,
  },

  imageTitle: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 12,
    color: COLORS.black,
  },

  imageText: {
    fontFamily: 'Inter_500Medium',
    fontSize: 11,
    color: COLORS.sage,
    marginTop: 2,
  },

  removeImageBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: COLORS.white,
    alignItems: 'center',
    justifyContent: 'center',
  },

  inputOuter: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: Platform.OS === 'ios' ? 82 : 76,
    paddingHorizontal: 14,
    paddingTop: 8,
    paddingBottom: 8,
    backgroundColor: 'rgba(233,233,224,0.96)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(218,217,207,0.85)',
    zIndex: 50,
    elevation: 50,
  },

  inputBar: {
    minHeight: 54,
    borderRadius: 28,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 8,
    paddingVertical: 6,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 8,
  },

  inputIconBtn: {
    width: 39,
    height: 39,
    borderRadius: 20,
    backgroundColor: COLORS.cream,
    alignItems: 'center',
    justifyContent: 'center',
  },

  input: {
    flex: 1,
    maxHeight: 96,
    fontFamily: 'Inter_600SemiBold',
    fontSize: 14,
    color: COLORS.black,
    paddingVertical: 8,
  },

  micBtn: {
    width: 41,
    height: 41,
    borderRadius: 21,
    backgroundColor: COLORS.earth,
    alignItems: 'center',
    justifyContent: 'center',
  },

  micBtnActive: {
    backgroundColor: COLORS.primary,
  },

  sendBtn: {
    width: 41,
    height: 41,
    borderRadius: 21,
    backgroundColor: COLORS.sage,
    alignItems: 'center',
    justifyContent: 'center',
  },

  sendBtnDisabled: {
    opacity: 0.65,
  },

  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingLeft: 8,
    paddingTop: 7,
  },

  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: COLORS.success,
  },

  statusDotBusy: {
    backgroundColor: COLORS.warning,
  },

  statusDotError: {
    backgroundColor: '#C84C4C',
  },

  statusText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 11,
    color: COLORS.sage,
  },
});