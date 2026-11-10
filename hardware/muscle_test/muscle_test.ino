/*
  muscle_test.ino  (one-channel signal check on A0 for the Arduino Serial Plotter)

  Open Tools > Serial Plotter at 115200 baud. Stay relaxed for the first 3 s
  (baseline), then squeeze. Every 50 ms it plots:
    signal     mean of A0 over the last 50 ms
    rest       baseline: mean signal over the first 3 s
    pass_line  max(3 x rest, rest + 30); a firm squeeze should go above it
    spread     max - min of A0 within the last 50 ms (large = noise or mains hum)
    top        fixed line at PLOT_TOP    } stop the plotter auto-zooming, so
    bottom     fixed line at PLOT_BOTTOM } read the axis numbers, not the shape
  Type r and press Enter in the plotter's message box to redo the baseline.

  Same wiring as emg_logger.ino: SIG -> 10k -> node -> 10k -> GND; node -> A0.
  Close the plotter before running the Python recorder.
*/

const unsigned long BAUD_RATE = 115200;
const byte SIGNAL_PIN = A0;
const unsigned long WINDOW_MS = 50;
const unsigned long READ_EVERY_US = 1000;   // about 50 readings per window
const unsigned long BASELINE_MS = 3000;
const long PASS_RATIO = 3;
const long PASS_MARGIN = 30;
const int PLOT_TOP = 1023;
const int PLOT_BOTTOM = 0;
const char REDO_BASELINE_KEY = 'r';

unsigned long windowStartMs = 0;
unsigned long lastReadUs = 0;
long windowSum = 0;
int windowCount = 0;
int windowMin = 1023;
int windowMax = 0;

unsigned long baselineStartMs = 0;
long baselineSum = 0;
long baselineWindows = 0;
long restLevel = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  startBaseline();
  startWindow();
}

void loop() {
  handleKeys();
  if (micros() - lastReadUs >= READ_EVERY_US) {
    lastReadUs = micros();
    addReading(analogRead(SIGNAL_PIN));
  }
  if (millis() - windowStartMs >= WINDOW_MS && windowCount > 0) {
    finishWindow();
    startWindow();
  }
}

// Forget the old baseline and measure a new one over the next BASELINE_MS.
void startBaseline() {
  baselineStartMs = millis();
  baselineSum = 0;
  baselineWindows = 0;
  restLevel = 0;
}

void startWindow() {
  windowStartMs = millis();
  windowSum = 0;
  windowCount = 0;
  windowMin = 1023;
  windowMax = 0;
}

void addReading(int reading) {
  windowSum += reading;
  windowCount++;
  if (reading < windowMin) windowMin = reading;
  if (reading > windowMax) windowMax = reading;
}

// Close the 50 ms window: update the baseline while it is running, then plot.
void finishWindow() {
  long signalLevel = windowSum / windowCount;
  if (millis() - baselineStartMs < BASELINE_MS) {
    baselineSum += signalLevel;
    baselineWindows++;
    restLevel = baselineSum / baselineWindows;
  }
  long passLine = max(PASS_RATIO * restLevel, restLevel + PASS_MARGIN);
  printPlotLine(signalLevel, passLine, windowMax - windowMin);
}

void printPlotLine(long signalLevel, long passLine, int spread) {
  Serial.print(F("signal:"));
  Serial.print(signalLevel);
  Serial.print(F(",rest:"));
  Serial.print(restLevel);
  Serial.print(F(",pass_line:"));
  Serial.print(passLine);
  Serial.print(F(",spread:"));
  Serial.print(spread);
  Serial.print(F(",top:"));
  Serial.print(PLOT_TOP);
  Serial.print(F(",bottom:"));
  Serial.println(PLOT_BOTTOM);
}

void handleKeys() {
  while (Serial.available() > 0) {
    char key = Serial.read();
    if (key == REDO_BASELINE_KEY || key == 'R') {
      startBaseline();
    }
  }
}
