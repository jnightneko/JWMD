// Motor principal (brazo)
const int IN1 = 8;
const int IN2 = 9;
const int ENA = 5; // Se deja por si se usa luego

// Motores de la cinta
const int IN3 = 6;
const int IN4 = 7;
const int ENB = 10; // velocidad de la cinta

int velocidadCinta = 130;
bool flag = false;

void setup() {
  Serial.begin(9600);

  // Configurar pines
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  Serial.println("Escribe 1 para mover el brazo, 0 para mantenerlo en reposo.");
}

void loop() {
  // Leer desde el monitor serial
  if (Serial.available() > 0) {
    char entrada = Serial.read();
    if (entrada == '1') {
      flag = true;
    } else if (entrada == '0') {
      flag = false;
    }
  }

  if (flag == true) {
    // Detener cinta
    detenerCinta();

    // Mover brazo a posición objetivo (derecha)
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    delay(1650);

    // Regresar brazo (izquierda)
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    delay(1600);

    // Detener brazo
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);

    // Reiniciar flag para esperar otra orden
    flag = false;
  } else {
    // Cinta: sigue avanzando
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    analogWrite(ENB, velocidadCinta);
  }

  delay(100);
}

void detenerCinta() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);
}