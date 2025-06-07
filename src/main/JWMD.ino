#define ESTADO_BUENO "BUENO"
#define ESTADO_MALO "MALO"

String inputString = "";
bool stringComplete = false;

void setup() {
  // inicializar el puerto serie
  Serial.begin(9600);
  inputString.reserve(200);
}

void loop() {
  serialEvent();

  if (stringComplete) {
    switch (inputString) 
    {
      case ESTADO_BUENO:
        botellaBuena();
        break;
      case ESTADO_MALO:
        botellaMala();
        break;
      default:
        break;
    }

    Serial.println(inputString);

    // limpiar el 'buffer'
    inputString = "";
    stringComplete = false;
  }
}

void botellaMala() {

}

void botellaBuena() {

}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;

    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
