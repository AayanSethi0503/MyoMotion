/*
  emg_logger.ino  (MyoMotion sEMG logger: 3 channels at 500 Hz, sync LED)

  Streams the three AD8226 envelope channels to the PC for
  scripts/semg/record_session.py. Upload this sketch, then CLOSE the Serial
  Monitor and Serial Plotter: only one program can use the port.

  Output, text at 250000 baud, one line each, "\r\n" endings:
    # emg_logger v1 rate_hz=500 channels=3   banner, once after each reset
    D,<t_us>,<ch1>,<ch2>,<ch3>               one sample; t_us = micros() when the
                                             sample was taken; chN = 0..1023
    E,<t_us>,LED_ON                          the sync LED just turned on; t_us exact

  Input: the character 'S' starts one 200 ms flash of the sync LED.

  micros() wraps to 0 every ~71.6 minutes; the Python recorder unwraps it.
  If the loop ever falls more than one period behind, the late samples are
  skipped and show up as a gap (missing samples) instead of a burst.

  Wiring (each channel; never connect SIG straight to the pin):
    SIG -> 10k -> node -> 10k -> GND rail; node -> A0 (ch1), A1 (ch2), A2 (ch3)
    both module GNDs -> GND rail -> Arduino GND; never +9 V or -9 V to the Arduino
  Sync LED: pin 8 -> 220 ohm -> LED long leg; LED short leg -> GND. Keep it in camera view.
*/

const unsigned long BAUD_RATE = 250000;
const unsigned long SAMPLE_PERIOD_US = 2000;  // 500 Hz
const byte CHANNEL_PINS[] = {A0, A1, A2};
const byte N_CHANNELS = sizeof(CHANNEL_PINS) / sizeof(CHANNEL_PINS[0]);
const byte SYNC_LED_PIN = 8;
const unsigned long LED_FLASH_US = 200000;    // 200 ms
const char SYNC_COMMAND = 'S';

unsigned long nextSampleUs = 0;
unsigned long ledOnSinceUs = 0;
bool ledIsOn = false;

void setup() {
  pinMode(SYNC_LED_PIN, OUTPUT);
  digitalWrite(SYNC_LED_PIN, LOW);
  Serial.begin(BAUD_RATE);
  Serial.print(F("# emg_logger v1 rate_hz="));
  Serial.print(1000000UL / SAMPLE_PERIOD_US);
  Serial.print(F(" channels="));
  Serial.println(N_CHANNELS);
  nextSampleUs = micros();
}

void loop() {
  if ((long)(micros() - nextSampleUs) >= 0) {
    sendSample();
    nextSampleUs += SAMPLE_PERIOD_US;
    skipLateSamples();
  }
  handleSyncCommand();
  turnLedOffWhenDone();
}

// Read every channel and send one "D,t_us,ch1,ch2,ch3" line.
void sendSample() {
  unsigned long sampleUs = micros();
  int readings[N_CHANNELS];
  for (byte channel = 0; channel < N_CHANNELS; channel++) {
    readings[channel] = analogRead(CHANNEL_PINS[channel]);
  }
  Serial.print('D');
  Serial.print(',');
  Serial.print(sampleUs);
  for (byte channel = 0; channel < N_CHANNELS; channel++) {
    Serial.print(',');
    Serial.print(readings[channel]);
  }
  Serial.println();
}

// If we are more than one whole period behind, restart the schedule from now.
void skipLateSamples() {
  unsigned long nowUs = micros();
  if ((long)(nowUs - nextSampleUs) >= (long)SAMPLE_PERIOD_US) {
    nextSampleUs = nowUs + SAMPLE_PERIOD_US;
  }
}

// On 'S': turn the LED on and report the exact micros() of that moment.
void handleSyncCommand() {
  while (Serial.available() > 0) {
    if (Serial.read() == SYNC_COMMAND && !ledIsOn) {
      digitalWrite(SYNC_LED_PIN, HIGH);
      ledOnSinceUs = micros();
      ledIsOn = true;
      Serial.print(F("E,"));
      Serial.print(ledOnSinceUs);
      Serial.println(F(",LED_ON"));
    }
  }
}

// Turn the LED off once the flash has lasted LED_FLASH_US (never blocks).
void turnLedOffWhenDone() {
  if (ledIsOn && micros() - ledOnSinceUs >= LED_FLASH_US) {
    digitalWrite(SYNC_LED_PIN, LOW);
    ledIsOn = false;
  }
}
