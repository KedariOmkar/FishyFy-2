#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

const char* ssid = "tesla";
const char* password = "elonmusk";

ESP8266WebServer server(80);  // Create a web server listening on port 80

// Define sensor pins
const int mq135Pin = A0;   // MQ135 sensor analog pin
const int mq9Pin = D1;     // MQ9 sensor digital pin
const int mq8Pin = D2;     // MQ8 sensor digital pin

void setup() {
  Serial.begin(115200);

  // Set pin modes
  pinMode(mq135Pin, INPUT);
  pinMode(mq9Pin, INPUT);
  pinMode(mq8Pin, INPUT);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Define routes
  server.on("/", HTTP_GET, [](){
    server.send(200, "text/plain", "Hello from Arduino!");
  });

  server.on("/check_status", HTTP_GET, [](){
    Serial.println("Request received: /check_status");
    String wifiInfo = getWiFiInfo();
    server.send(200, "application/json", wifiInfo);
  });

  server.on("/get_readings", HTTP_GET, [](){
    Serial.println("Request received: /get_readings");
    String readings = getGasSensorReadings();
    server.send(200, "application/json", readings);
  });

  // Start the server
  server.begin();
  Serial.println("Server started");
}

void loop() {
  server.handleClient(); // Handle client requests
}

String getWiFiInfo() {
  String json = "{";
  json += "\"ip_address\": \"" + WiFi.localIP().toString() + "\",";
  json += "\"ssid\": \"" + String(ssid) + "\",";
  json += "\"password\": \"" + String(password) + "\"";
  json += "}";

  Serial.println("WiFi info: " + json);
  return json;
}

String getGasSensorReadings() {
  // Variables to store accumulated sensor values
  long totalMQ135Value = 0;
  int numReadings = 0;

  unsigned long startMillis = millis();
  while (millis() - startMillis < 15000) {  // 15 seconds
    totalMQ135Value += analogRead(mq135Pin);
    numReadings++;
    delay(100);  // Small delay between readings
  }

  // Calculate average sensor values
  int avgMQ135Value = totalMQ135Value / numReadings;

  // Placeholder conversion formulas for gas concentrations (PPM)
  // Adjust these based on your calibration
  float mq135PPM = avgMQ135Value * 0.1; // Modify based on calibration

  // Read digital sensors
  bool mq9State = digitalRead(mq9Pin);
  bool mq8State = digitalRead(mq8Pin);

  // Construct JSON response
  String json = "{";
  json += "\"MQ135_value\": " + String(avgMQ135Value) + ",";
  json += "\"MQ135_ppm\": " + String(mq135PPM, 2) + ",";
  json += "\"MQ9_state\": " + String(mq9State) + ",";
  json += "\"MQ8_state\": " + String(mq8State);
  json += "}";

  Serial.println("Gas sensor readings: " + json);
  return json;

}
