/*
 * SmartClassroom AI — ESP32-S3 OLED Client Node
 * Phase 6 Implementation: Improved ESP32-S3 Integration
 *
 * Features:
 *  - Persistent, Configurable API Server Address via Preferences (NVS) & Serial Commands
 *  - Non-blocking Wi-Fi Reconnection Logic
 *  - High-information SH1106G OLED Dashboard (Occupancy, Drowsiness, Liveness, Target IP)
 *  - Visual Inverted Fatigue Alert State
 *  - Descriptive Connection & HTTP Error Diagnostics
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Preferences.h>

// ---------------- I2C & OLED PIN CONFIGURATION ----------------
#define OLED_SDA 8
#define OLED_SCL 9

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDRESS 0x3C

Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ---------------- DEFAULT CONFIGURATION ----------------
const char* DEFAULT_SSID = "Daham";
const char* DEFAULT_PASS = "daham123";
const char* DEFAULT_SERVER_IP = "172.20.10.7";
const int   DEFAULT_SERVER_PORT = 5000;

// ---------------- CONFIGURATION STATE (NVS) ----------------
Preferences prefs;
String currentSsid;
String currentPass;
String currentServerIp;
int currentServerPort;
String apiUrl;

// Polling interval (3 seconds)
unsigned long lastRequest = 0;
const unsigned long REQUEST_INTERVAL = 3000;

// Wi-Fi reconnection timer
unsigned long lastWiFiReconnect = 0;
const unsigned long WIFI_RETRY_INTERVAL = 5000;

// ---------------- HELPER: BUILD API URL ----------------
void updateApiUrl() {
  apiUrl = "http://" + currentServerIp + ":" + String(currentServerPort) + "/classroom";
  Serial.println("[CONFIG] Active Target API URL: " + apiUrl);
}

// ---------------- HELPER: LOAD CONFIG FROM FLASH ----------------
void loadConfiguration() {
  prefs.begin("classroom_cfg", false);
  currentSsid = prefs.getString("ssid", DEFAULT_SSID);
  currentPass = prefs.getString("pass", DEFAULT_PASS);
  currentServerIp = prefs.getString("server_ip", DEFAULT_SERVER_IP);
  currentServerPort = prefs.getInt("server_port", DEFAULT_SERVER_PORT);
  prefs.end();

  updateApiUrl();
}

// ---------------- HELPER: SAVE SERVER IP ----------------
void saveServerIp(const String& newIp, int newPort = 5000) {
  prefs.begin("classroom_cfg", false);
  prefs.putString("server_ip", newIp);
  prefs.putInt("server_port", newPort);
  prefs.end();

  currentServerIp = newIp;
  currentServerPort = newPort;
  updateApiUrl();
  Serial.println("[CONFIG] New server IP saved to NVS: " + currentServerIp);
}

// ---------------- HELPER: RESET CONFIGURATION ----------------
void resetConfiguration() {
  prefs.begin("classroom_cfg", false);
  prefs.clear();
  prefs.end();

  currentSsid = DEFAULT_SSID;
  currentPass = DEFAULT_PASS;
  currentServerIp = DEFAULT_SERVER_IP;
  currentServerPort = DEFAULT_SERVER_PORT;
  updateApiUrl();
  Serial.println("[CONFIG] Preferences reset to default settings.");
}

// ---------------- OLED MESSAGE HELPER ----------------
void showStatusScreen(const char* title, const char* line1, const char* line2 = "", const char* line3 = "") {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);

  // Header
  display.setCursor(0, 0);
  display.println(title);
  display.drawLine(0, 10, 127, 10, SH110X_WHITE);

  // Content
  display.setCursor(0, 18);
  display.println(line1);

  if (strlen(line2) > 0) {
    display.setCursor(0, 32);
    display.println(line2);
  }

  if (strlen(line3) > 0) {
    display.setCursor(0, 48);
    display.println(line3);
  }

  display.display();
}

// ---------------- OLED DASHBOARD SCREEN ----------------
void renderDashboard(int occupancy, int drowsyCount, bool isLive, const String& statusText) {
  display.clearDisplay();
  display.setTextSize(1);

  bool hasAlert = (drowsyCount > 0);

  // Header Bar (Inverted white banner when drowsiness alert is active)
  if (hasAlert) {
    display.fillRect(0, 0, 128, 12, SH110X_WHITE);
    display.setTextColor(SH110X_BLACK, SH110X_WHITE);
    display.setCursor(6, 2);
    display.print("! FATIGUE ALERT !");
  } else {
    display.setTextColor(SH110X_WHITE);
    display.setCursor(0, 0);
    display.print("SMART CLASSROOM");
    display.drawLine(0, 11, 127, 11, SH110X_WHITE);
  }

  // Row 1: Occupancy
  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 16);
  display.print("Students : ");
  display.setTextSize(1);
  display.print(occupancy);

  // Row 2: Drowsiness
  display.setCursor(0, 29);
  display.print("Drowsy   : ");
  display.print(drowsyCount);
  if (hasAlert) {
    display.print(" [WARN]");
  } else {
    display.print(" [OK]");
  }

  // Row 3: Liveness & Status
  display.setCursor(0, 42);
  display.print("AI Feed  : ");
  if (isLive) {
    display.print("LIVE");
  } else {
    display.print("STALE");
  }

  // Bottom Divider & Host info
  display.drawLine(0, 53, 127, 53, SH110X_WHITE);
  display.setCursor(0, 56);
  display.print("Host: ");
  display.print(currentServerIp);

  display.display();
}

// ---------------- OLED HTTP ERROR SCREEN ----------------
void renderHttpError(int httpCode) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);

  // Inverted error banner
  display.fillRect(0, 0, 128, 12, SH110X_WHITE);
  display.setTextColor(SH110X_BLACK, SH110X_WHITE);
  display.setCursor(10, 2);
  display.print("API UNAVAILABLE");

  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 17);
  display.printf("Error Code: %d", httpCode);

  display.setCursor(0, 30);
  if (httpCode == -1) {
    display.println("Connection Refused");
    display.setCursor(0, 41);
    display.println("Check PC Firewall/IP");
  } else if (httpCode == -11) {
    display.println("Read Timeout");
  } else if (httpCode == 404) {
    display.println("Route /classroom 404");
  } else {
    display.println("HTTP Request Failed");
  }

  display.drawLine(0, 52, 127, 52, SH110X_WHITE);
  display.setCursor(0, 55);
  display.print("Target: ");
  display.print(currentServerIp);

  display.display();
}

// ---------------- SERIAL COMMAND PARSER ----------------
// Allows changing IP dynamically: e.g., "SET_IP 192.168.1.10" or "GET_CONFIG"
void checkSerialCommands() {
  if (!Serial.available()) return;

  String input = Serial.readStringUntil('\n');
  input.trim();

  if (input.startsWith("SET_IP ")) {
    String newIp = input.substring(7);
    newIp.trim();
    if (newIp.length() > 0) {
      saveServerIp(newIp, currentServerPort);
      Serial.println("[SUCCESS] Server IP updated to: " + newIp);
      showStatusScreen("CONFIG UPDATED", ("IP: " + newIp).c_str(), "Connecting...");
      delay(1200);
    }
  } else if (input.startsWith("SET_PORT ")) {
    int newPort = input.substring(9).toInt();
    if (newPort > 0) {
      saveServerIp(currentServerIp, newPort);
      Serial.printf("[SUCCESS] Server port updated to: %d\n", newPort);
    }
  } else if (input.equalsIgnoreCase("GET_CONFIG")) {
    Serial.println("--- Current ESP32 Configuration ---");
    Serial.println("SSID: " + currentSsid);
    Serial.println("Server IP: " + currentServerIp);
    Serial.printf("Server Port: %d\n", currentServerPort);
    Serial.println("API URL: " + apiUrl);
    Serial.println("WiFi Status: " + String(WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected"));
    Serial.println("ESP32 Local IP: " + WiFi.localIP().toString());
    Serial.println("-----------------------------------");
  } else if (input.equalsIgnoreCase("RESET_CONFIG")) {
    resetConfiguration();
  } else if (input.equalsIgnoreCase("HELP")) {
    Serial.println("Commands:");
    Serial.println("  SET_IP <ip_address>   (e.g., SET_IP 192.168.1.50)");
    Serial.println("  SET_PORT <port>       (e.g., SET_PORT 5000)");
    Serial.println("  GET_CONFIG            (Display current configuration)");
    Serial.println("  RESET_CONFIG          (Reset to default 172.20.10.7)");
  }
}

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n======================================");
  Serial.println(" SmartClassroom AI - ESP32-S3 Client");
  Serial.println(" Type 'HELP' in Serial to view commands");
  Serial.println("======================================");

  // Initialize I2C and Display
  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(OLED_ADDRESS, true)) {
    Serial.println("[ERROR] SH1106G OLED initialization failed!");
    while (true) delay(100);
  }

  display.clearDisplay();
  display.display();

  // Load persistent configuration from NVS
  loadConfiguration();

  showStatusScreen("SMART CLASSROOM", "Connecting WiFi...", currentSsid.c_str());

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  WiFi.begin(currentSsid.c_str(), currentPass.c_str());

  Serial.print("Connecting to Wi-Fi: " + currentSsid);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 25) {
    delay(400);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected! Local IP: " + WiFi.localIP().toString());
    showStatusScreen("WIFI CONNECTED", ("IP: " + WiFi.localIP().toString()).c_str(), ("Target: " + currentServerIp).c_str());
  } else {
    Serial.println("\n[WIFI] Initial connection failed. Will retry in loop.");
    showStatusScreen("WIFI FAILED", "Will retry in background", ("SSID: " + currentSsid).c_str());
  }

  delay(1200);
}

// ---------------- MAIN LOOP ----------------
void loop() {
  // 1. Check for incoming Serial commands (dynamic IP changes)
  checkSerialCommands();

  // 2. Handle Wi-Fi Reconnection Logic
  if (WiFi.status() != WL_CONNECTED) {
    unsigned long now = millis();
    if (now - lastWiFiReconnect >= WIFI_RETRY_INTERVAL) {
      lastWiFiReconnect = now;
      Serial.println("[WIFI] Reconnecting to: " + currentSsid);
      showStatusScreen("WIFI DISCONNECTED", "Reconnecting...", currentSsid.c_str());
      WiFi.disconnect();
      WiFi.reconnect();
    }
    delay(100);
    return;
  }

  // 3. Poll Flask Server at configured interval
  unsigned long now = millis();
  if (now - lastRequest >= REQUEST_INTERVAL) {
    lastRequest = now;

    HTTPClient http;
    http.setTimeout(3000);
    http.begin(apiUrl);

    int httpCode = http.GET();

    if (httpCode == HTTP_CODE_OK) {
      String payload = http.getString();

      JsonDocument doc;
      DeserializationError error = deserializeJson(doc, payload);

      if (!error) {
        int occupancy = doc["occupancy"] | 0;
        int drowsyCount = doc["drowsy_count"] | 0;
        bool isLive = doc["is_live"] | false;
        String statusText = doc["status"] | "waiting";

        // Render high-information dashboard on OLED
        renderDashboard(occupancy, drowsyCount, isLive, statusText);

        Serial.printf("[HTTP 200] Occupancy: %d | Drowsy: %d | Live: %s\n",
                      occupancy, drowsyCount, isLive ? "YES" : "NO");
      } else {
        showStatusScreen("JSON ERROR", "Failed to parse API data", payload.substring(0, 16).c_str());
        Serial.println("[ERROR] JSON deserialization failed.");
      }
    } else {
      // Render descriptive HTTP error screen
      renderHttpError(httpCode);
      Serial.printf("[HTTP ERROR] %d while querying %s\n", httpCode, apiUrl.c_str());
    }

    http.end();
  }
}
