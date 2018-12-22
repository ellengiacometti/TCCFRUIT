
#include <Servo.h> //INCLUSÃO DA BIBLIOTECA NECESSÁRIA
// defines pins numbers
const int trigPin = 11;// PINO TRIG
const int echoPin = 10; //PINO ECHO 
const int pinoServo = 9 ; //PINO SERVO 
Servo s; //OBJETO DO TIPO SERVO
int pos; //POSIÇÃO DO SERVO

// defines variables
long duration;
int distance;
void setup() {
  Serial.begin(9600); // Starts the serial communication
  pinMode(trigPin, OUTPUT); // ASSOCIA O trigPin como SAIDA
  pinMode(echoPin, INPUT); // ASSOCIA O echoPin como ENTRADA
  s.attach(pinoServo); //ASSOCIAÇÃO DO PINO DIG AO SERVO
  s.write(90); //INICIA O MOTOR NA POSIÇÃO 0º
 
}

void loop() {
digitalWrite(trigPin, LOW);
delayMicroseconds(1000);
// Sets the trigPin on HIGH state for 10 micro seconds
digitalWrite(trigPin, HIGH);
delayMicroseconds(1000);
digitalWrite(trigPin, LOW);

// Reads the echoPin, returns the sound wave travel time in microseconds
duration = pulseIn(echoPin, HIGH);
digitalWrite(trigPin, LOW);
delayMicroseconds(1000);

// Calculating the distance
distance= duration*0.034/2;

// Prints the distance on the Serial Monitor
Serial.print("Distancia: ");
Serial.println(distance);

  if(distance <10){
    for(pos = 90; pos < 170; pos++){ //PARA "pos" IGUAL A 0, ENQUANTO "pos" MENOR QUE 180, INCREMENTA "pos"
    s.write(pos); //ESCREVE O VALOR DA POSIÇÃO QUE O SERVO DEVE GIRAR
    delay(30); //INTERVALO DE 30 MILISSEGUNDOS
    }
    delay(1000);
    Serial.print("Abaxei");
    delay(8000);
    Serial.print("Levantei");
    s.write(90);
    delay(1000);
   }
  else {
    s.write(90);
    Serial.print("Tá Levantado");
 
     }
  }
