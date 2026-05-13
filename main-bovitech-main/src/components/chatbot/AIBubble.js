import React, { memo, useEffect, useRef } from 'react';
import { Animated, Easing, Image, StyleSheet, View } from 'react-native';

const aiCowImage = require('../../../assets/ai-cow.jpg');

const COLORS = {
  primary: '#2E412D',
  cream: '#F6F6EC',
  discussion: '#E9E9E0',
};

const AIBubble = memo(function AIBubble({ size = 154, active = true, compact = false }) {
  const pulse = useRef(new Animated.Value(0)).current;
  const float = useRef(new Animated.Value(0)).current;
  const glow = useRef(new Animated.Value(0)).current;
  const rotate = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (!active) return;

    const pulseLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          toValue: 1,
          duration: 1400,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(pulse, {
          toValue: 0,
          duration: 1400,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ])
    );

    const floatLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(float, {
          toValue: 1,
          duration: 1900,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(float, {
          toValue: 0,
          duration: 1900,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ])
    );

    const glowLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(glow, {
          toValue: 1,
          duration: 1600,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(glow, {
          toValue: 0,
          duration: 1600,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ])
    );

    const rotateLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(rotate, {
          toValue: 1,
          duration: 3200,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(rotate, {
          toValue: 0,
          duration: 3200,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ])
    );

    pulseLoop.start();
    floatLoop.start();
    glowLoop.start();
    rotateLoop.start();

    return () => {
      pulseLoop.stop();
      floatLoop.stop();
      glowLoop.stop();
      rotateLoop.stop();
    };
  }, [active, pulse, float, glow, rotate]);

  const imageSize = compact ? size * 0.88 : size * 0.94;

  const scale = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: [1, compact ? 1.035 : 1.055],
  });

  const translateY = float.interpolate({
    inputRange: [0, 1],
    outputRange: [0, compact ? -2 : -7],
  });

  const haloScale = glow.interpolate({
    inputRange: [0, 1],
    outputRange: [1, 1.08],
  });

  const haloOpacity = glow.interpolate({
    inputRange: [0, 1],
    outputRange: [0.36, 0.72],
  });

  const rotateZ = rotate.interpolate({
    inputRange: [0, 1],
    outputRange: ['-1.4deg', '1.4deg'],
  });

  return (
    <View style={[styles.wrapper, { width: size, height: size }]}>
      {!compact && (
        <>
          <Animated.View
            style={[
              styles.haloOuter,
              {
                width: size,
                height: size,
                borderRadius: size / 2,
                opacity: haloOpacity,
                transform: [{ scale: haloScale }],
              },
            ]}
          />

          <View
            style={[
              styles.haloInner,
              {
                width: size * 0.82,
                height: size * 0.82,
                borderRadius: (size * 0.82) / 2,
              },
            ]}
          />
        </>
      )}

      <Animated.View
        style={[
          styles.imageWrap,
          compact && styles.imageWrapCompact,
          {
            width: imageSize,
            height: imageSize,
            borderRadius: imageSize / 2,
            transform: [{ translateY }, { scale }, { rotate: rotateZ }],
          },
        ]}
      >
        <Image source={aiCowImage} style={styles.image} resizeMode="cover" />
      </Animated.View>

      {!compact && <View style={styles.floorShadow} />}
    </View>
  );
});

const styles = StyleSheet.create({
  wrapper: {
    alignItems: 'center',
    justifyContent: 'center',
  },

  haloOuter: {
    position: 'absolute',
    backgroundColor: 'rgba(46,65,45,0.08)',
  },

  haloInner: {
    position: 'absolute',
    backgroundColor: 'rgba(46,65,45,0.07)',
  },

  imageWrap: {
    backgroundColor: COLORS.discussion,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.58)',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.28,
    shadowRadius: 28,
    elevation: 15,
  },

  imageWrapCompact: {
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.16,
    shadowRadius: 8,
    elevation: 4,
  },

  image: {
    width: '100%',
    height: '100%',
    backgroundColor: COLORS.discussion,
  },

  floorShadow: {
    position: 'absolute',
    bottom: 18,
    width: 92,
    height: 18,
    borderRadius: 30,
    backgroundColor: 'rgba(46,65,45,0.16)',
    transform: [{ scaleX: 1.25 }],
  },
});

export default AIBubble;