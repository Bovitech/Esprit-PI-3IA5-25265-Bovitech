const path = require('path');

// Charge .env depuis la racine du projet Expo (bovitech-main/)
require('dotenv').config({ path: path.join(__dirname, '.env') });

const appJson = require('./app.json');

const apiBaseUrl = process.env.EXPO_PUBLIC_API_BASE_URL || '';
const chatbotBaseUrl = process.env.EXPO_PUBLIC_CHATBOT_BASE_URL || '';

module.exports = {
  expo: {
    ...appJson.expo,
    ios: {
      ...(appJson.expo.ios || {}),
      infoPlist: {
        ...(appJson.expo.ios?.infoPlist || {}),
        // HTTP vers le PC sur le LAN (sinon iOS bloque souvent)
        NSAppTransportSecurity: {
          NSAllowsLocalNetworking: true,
          NSAllowsArbitraryLoads: true,
        },
        // Pour ouvrir Google Maps / Plans depuis Linking.canOpenURL
        LSApplicationQueriesSchemes: [
          ...(appJson.expo.ios?.infoPlist?.LSApplicationQueriesSchemes || []),
          'comgooglemaps',
          'maps',
        ],
      },
    },
    extra: {
      ...(appJson.expo.extra || {}),
      apiBaseUrl,
      chatbotBaseUrl,
    },
  },
};
