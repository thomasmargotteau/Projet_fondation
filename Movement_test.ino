#define BLYNK_TEMPLATE_ID "TMPL5cesKbWRF"
#define BLYNK_TEMPLATE_NAME "Mouvement Foundation"
#define BLYNK_AUTH_TOKEN "iT3oafW0zhvbXk4Dak_CvZCOePFJ0G_y"
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

int dir;

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
  servo3.write(s2);
  Blynk.virtualWrite(V5, s2);
} 

//-------------------Control driver motor-------------------
// останов
void stop(void) {     
   analogWrite(5, 0);     
     analogWrite(4, 0); 
}  
// вперед 
void forward(void) {
     analogWrite(5, 255); analogWrite(4, 255);
     digitalWrite(0, HIGH);digitalWrite(2, HIGH); 
}  
// назад 
void backward(void) {
     analogWrite(5, 255);analogWrite(4, 255);
     digitalWrite(0, LOW);digitalWrite(2, LOW); 
}   
// влево
void left(void) {
     analogWrite(5, 255);analogWrite(4, 255);
     digitalWrite(0, LOW);digitalWrite(2, HIGH);
}   
// вправо
void right(void) {
     analogWrite(5, 255);analogWrite(4, 255);
     digitalWrite(0, HIGH); digitalWrite(2, LOW); 
}   
//--------------------------------------------------------------

//Two motors controlled with a joystick connected to the virtual pin V2
BLYNK_WRITE(V6) {
  int x = param[0].asInt();
  int y = param[1].asInt();

       if (x<20) {dir=1;}
  else if (x>235) {dir=2;}
  else if (y>235) {dir=3;}
  else if (y<20) {dir=4;}
  else if (x>=20 && x<=235 && y>=20 && y<=235) {dir=5;}
         // выбор для кнопок         
         switch (dir)   {
             case 1:  left();
                  break;             
             case 2:  right();
                 break;
             case 3:  forward();
                 break;
             case 4: backward();
                break;             
             case 5:  stop(); 
                break;         } 
  
}

void setup()
{
  Serial.begin(9600);
  servo1.attach(D6); 
  servo2.attach(D7); 
  servo3.attach(D8);

  Blynk.begin(auth, ssid, pass);
 
  pinMode(5, OUTPUT); // motor A speed
  pinMode(4, OUTPUT); // motor B speed
  pinMode(0, OUTPUT); //  motor A direction
  pinMode(2, OUTPUT); //  motor B direction
}

void loop() 
{
  Blynk.run();
}
