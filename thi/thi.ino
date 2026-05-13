/**
 * DHT22 + HTTP → API BoviTech (model_http_api.py :8008) — ESP32 uniquement.
 *
 * Ouvrez le fichier thi_esp32_http.ino dans ce dossier (un seul .ino ici).
 * Le dossier parent contient thi.ino (autre sketch) : ne pas mélanger les deux dans le même dossier.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

const char* WIFI_SSID = "Redmi 14C";
const char* WIFI_PASS = "maram2711";
const char* API_HOST = "192.168.28.189";
const int API_PORT = 8008;

void setup() {
  Serial.begin(115200);
  delay(500);
  dht.begin();
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connexion WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("OK — IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi perdu, reconnexion…");
    WiFi.reconnect();
    delay(2000);
    return;
  }

  float RH = dht.readHumidity();
  float T = dht.readTemperature();
  if (isnan(RH) || isnan(T)) {
    Serial.println("Erreur DHT22");
    delay(2000);
    return;
  }

  String body = "{\"temp_c\":";
  body += String(T, 2);
  body += ",\"humidity\":";
  body += String(RH, 2);
  body += "}";

  String url = "http://";
  url += API_HOST;
  url += ":";
  url += String(API_PORT);
  url += "/barn_sensor";

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json; charset=utf-8");
  int code = http.POST(body);
  Serial.print("POST /barn_sensor -> ");
  Serial.println(code);
  if (code < 0) {
    Serial.println(http.errorToString(code));
  }
  http.end();

  delay(2000);
}
