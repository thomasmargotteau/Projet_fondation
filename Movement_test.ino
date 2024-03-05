#define BLYNK_TEMPLATE_ID "TMPL5ELMfSxQz"
#define BLYNK_TEMPLATE_NAME "Servo with esp8266"
#define BLYNK_AUTH_TOKEN "iE4BMMTaKI-ACpisHo-Km-fpsihmWLNX"
#define BLYNK_PRINT Serial

#ifndef BlynkSimpleEsp8266_h
#define BlynkSimpleEsp8266_h

#ifndef ESP8266
#error This code is intended to run on the ESP8266 platform! Please check your Tools->Board setting.
#endif

#include <version.h>

#if ESP_SDK_VERSION_NUMBER < 0x020200
#error Please update your ESP8266 Arduino Core
#endif

#include <BlynkApiArduino.h>
#include <Blynk/BlynkProtocol.h>
#include <Adapters/BlynkArduinoClient.h>
#include <ESP8266WiFi.h>

class BlynkWifi
    : public BlynkProtocol<BlynkArduinoClient>
{
    typedef BlynkProtocol<BlynkArduinoClient> Base;
public:
    BlynkWifi(BlynkArduinoClient& transp)
        : Base(transp)
    {}

    void connectWiFi(const char* ssid, const char* pass)
    {
        BLYNK_LOG2(BLYNK_F("Connecting to "), ssid);
        WiFi.mode(WIFI_STA);
        if (WiFi.status() != WL_CONNECTED) {
            if (pass && strlen(pass)) {
                WiFi.begin(ssid, pass);
            } else {
                WiFi.begin(ssid);
            }
        }
        while (WiFi.status() != WL_CONNECTED) {
            BlynkDelay(500);
        }
        BLYNK_LOG1(BLYNK_F("Connected to WiFi"));

        IPAddress myip = WiFi.localIP();
        (void)myip; // Eliminate warnings about unused myip
        BLYNK_LOG_IP("IP: ", myip);
    }

    void config(const char* auth,
                const char* domain = BLYNK_DEFAULT_DOMAIN,
                uint16_t    port   = BLYNK_DEFAULT_PORT)
    {
        Base::begin(auth);
        this->conn.begin(domain, port);
    }

    void config(const char* auth,
                IPAddress   ip,
                uint16_t    port = BLYNK_DEFAULT_PORT)
    {
        Base::begin(auth);
        this->conn.begin(ip, port);
    }

    void begin(const char* auth,
               const char* ssid,
               const char* pass,
               const char* domain = BLYNK_DEFAULT_DOMAIN,
               uint16_t    port   = BLYNK_DEFAULT_PORT)
    {
        connectWiFi(ssid, pass);
        config(auth, domain, port);
        while(this->connect() != true) {}
    }

    void begin(const char* auth,
               const char* ssid,
               const char* pass,
               IPAddress   ip,
               uint16_t    port   = BLYNK_DEFAULT_PORT)
    {
        connectWiFi(ssid, pass);
        config(auth, ip, port);
        while(this->connect() != true) {}
    }

};

#if !defined(NO_GLOBAL_INSTANCES) && !defined(NO_GLOBAL_BLYNK)
  static WiFiClient _blynkWifiClient;
  static BlynkArduinoClient _blynkTransport(_blynkWifiClient);
  BlynkWifi Blynk(_blynkTransport);
#else
  extern BlynkWifi Blynk;
#endif

#include <BlynkWidgets.h>

#endif

#define RightMotorSpeed 5
#define RightMotorDir   0 
#define LeftMotorSpeed  4
#define LeftMotorDir    2

#include <ESP8266WiFi.h>
#include <Servo.h>
#include <BlynkSimpleEsp8266.h>
#include <SPI.h>

char auth[] = BLYNK_AUTH_TOKEN;
char ssid[] = "S10 Arthuuuuur";
char pass[] = "mgefhyff";

int minRange = 312;
int maxRange = 712;

int minSpeed = 450;
int maxSpeed = 1020;
int noSpeed = 0;

Servo servo1, servo2, servo3;

BLYNK_WRITE(V0)
{
  int s0 = param.asInt(); 
  servo1.write(s0);
  Blynk.virtualWrite(V2, s0);
} 

BLYNK_WRITE(V1)
{
  int s1 = param.asInt(); 
  servo2.write(s1);
  Blynk.virtualWrite(V3, s1);
} 

BLYNK_WRITE(V4)
{
  int s2 = param.asInt(); 
  servo2.write(s2);
  Blynk.virtualWrite(V5, s2);
} 

BLYNK_WRITE(V6)
{
  int x = param[0].asInt();
  int y = param[1].asInt();
  moveControl(x,y);
}

//-------------------Control driver motor-------------------
void moveControl(int x, int y)
{

  if(y >= maxRange && x >= minRange && x <= maxRange) //zataci R
  {
    digitalWrite(RightMotorDir,HIGH); 
    digitalWrite(LeftMotorDir,HIGH);
    analogWrite(RightMotorSpeed,maxSpeed);
    analogWrite(LeftMotorSpeed,maxSpeed);
  }
 
  // move forward right
  else if(x >= maxRange && y >= maxRange)   //zataci R
  {
    digitalWrite(RightMotorDir,HIGH);
    digitalWrite(LeftMotorDir,HIGH);
   analogWrite(RightMotorSpeed,minSpeed);
    analogWrite(LeftMotorSpeed,maxSpeed);
  }

  // move forward left
  else if(x <= minRange && y >= maxRange)
  {
    digitalWrite(RightMotorDir,HIGH);
    digitalWrite(LeftMotorDir,HIGH);
    analogWrite(RightMotorSpeed,maxSpeed);
    analogWrite(LeftMotorSpeed,minSpeed);
  }

  // neutral zone
  else if(y < maxRange && y > minRange && x < maxRange && x > minRange)
  {
    analogWrite(RightMotorSpeed,noSpeed);
    analogWrite(LeftMotorSpeed,noSpeed);
  }

 // move back
  else if(y <= minRange && x >= minRange && x <= maxRange)
  {
    digitalWrite(RightMotorDir,LOW);
    digitalWrite(LeftMotorDir,LOW);
   analogWrite(RightMotorSpeed,maxSpeed);
    analogWrite(LeftMotorSpeed,maxSpeed);
  }

  // move back and right
 else if(y <= minRange && x <= minRange)
  {
   digitalWrite(RightMotorDir,LOW);
    digitalWrite(LeftMotorDir,LOW);
    analogWrite(RightMotorSpeed,minSpeed);
    analogWrite(LeftMotorSpeed,maxSpeed); 
  }

  // move back and left
  else if(y <= minRange && x >= maxRange)
  {
    digitalWrite(RightMotorDir,LOW);
    digitalWrite(LeftMotorDir,LOW);
    analogWrite(RightMotorSpeed,maxSpeed);
    analogWrite(LeftMotorSpeed,minSpeed);
  }
}
//--------------------------------------------------------------

void setup()
{
  Serial.begin(9600);
  servo1.attach(D6); 
  servo2.attach(D7); 
  servo3.attach(D8);

  Blynk.begin(auth, ssid, pass);
  pinMode(RightMotorSpeed, OUTPUT);
  pinMode(LeftMotorSpeed, OUTPUT);
  pinMode(RightMotorDir, OUTPUT);
  pinMode(LeftMotorDir, OUTPUT);
 
  digitalWrite(RightMotorSpeed, LOW);
  digitalWrite(LeftMotorSpeed, LOW);
  digitalWrite(RightMotorDir, HIGH);
  digitalWrite(LeftMotorDir,HIGH);
 
}

void loop() 
{
  Blynk.run();
}