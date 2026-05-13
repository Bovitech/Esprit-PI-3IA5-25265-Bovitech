# 🐄 BoviTech — Application Mobile

Application de gestion de troupeau bovin développée avec **React Native + Expo**.

---

## 📁 Structure du projet

```
bovitech/
├── App.js                          ← Point d'entrée
├── package.json
└── src/
    ├── theme/
    │   ├── colors.js               ← Palette de couleurs
    │   └── mockData.js             ← Données simulées
    ├── components/
    │   └── UIComponents.js         ← Badge, Card, ProgressBar...
    ├── navigation/
    │   └── AppNavigator.js         ← Navigation par onglets
    └── screens/
        ├── HomeScreen.js           ← Dashboard + alertes
        ├── HerdScreen.js           ← Liste des vaches
        ├── GPSScreen.js            ← Carte GPS
        └── HealthScreen.js         ← Santé & IA
```

---

## 🚀 Installation et lancement

### 1. Prérequis
- Node.js 18+
- npm ou yarn
- Expo Go sur ton téléphone Android (depuis Play Store)

### 2. Installation
```bash
# Cloner ou créer le projet
npx create-expo-app bovitech --template blank
cd bovitech

# Copier tous les fichiers du projet dans ce dossier

# Installer les dépendances
npm install @react-navigation/native @react-navigation/bottom-tabs
npm install react-native-screens react-native-safe-area-context
```

### 3. Lancer l'app
```bash
npx expo start
```

Scanne le QR code avec **Expo Go** sur ton téléphone Android.

---

## 📱 Écrans de l'application

| Écran | Description |
|-------|-------------|
| 🏠 **Accueil** | Dashboard principal avec alertes, graphique lait, statut troupeau |
| 🐄 **Troupeau** | Liste des vaches avec détails (appuie sur une vache) |
| 🗺️ **GPS** | Carte de localisation, alertes hors-zone |
| ❤️ **Santé IA** | Prédictions maladies, comportement, événements |

---

## 🎨 Design System

Couleurs principales :
- **Vert** `#3B6D11` — couleur principale / bonne santé
- **Amber** `#BA7517` — avertissements
- **Rouge** `#A32D2D` — alertes urgentes
- **Teal** `#0F6E56` — informations / gestation

---

## 🔧 Prochaines étapes (Backend)

1. **API REST** : Node.js + Express ou FastAPI (Python)
2. **Base de données** : PostgreSQL pour les données historiques
3. **Temps réel** : Firebase Realtime DB ou WebSockets pour les alertes live
4. **IoT** : MQTT broker pour recevoir les données des colliers
5. **IA** : TensorFlow Lite pour les prédictions en local sur l'app

---

## 📦 Dépendances principales

```json
{
  "@react-navigation/native": "^6.x",
  "@react-navigation/bottom-tabs": "^6.x",
  "react-native-screens": "^3.x",
  "react-native-safe-area-context": "^4.x"
}
```
