// Samples the QIDK main supply rail (PMIC ADC channel pm8550 vph_pwr, microvolts) every <period_ms>
// for <seconds> and prints mean / std / count, plus the SoC thermistor at start and end.
// The rail sags in proportion to the current the board draws, so (idle mean - load mean) is a relative
// measure of extra power. Needs root (run with su).
//   vphmon <seconds> [period_ms=20]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const char* VPH = "/sys/bus/iio/devices/iio:device0/in_voltage_pm8550_vph_pwr_input";
static const char* MSM = "/sys/bus/iio/devices/iio:device0/in_temp_pm8550_msm_therm_input";

static long rd(const char* p) {
  FILE* f = fopen(p, "r");
  if (!f) return -1;
  long v = -1;
  if (fscanf(f, "%ld", &v) != 1) v = -1;
  fclose(f);
  return v;
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  double secs = argc > 1 ? atof(argv[1]) : 10;
  int period = argc > 2 ? atoi(argv[2]) : 20;
  long t0c = rd(MSM);
  double t0 = now(), s = 0, s2 = 0;
  long n = 0;
  struct timespec sl = {0, period * 1000000L};
  while (now() - t0 < secs) {
    long v = rd(VPH);
    if (v > 0) { s += v; s2 += (double)v * v; n++; }
    nanosleep(&sl, NULL);
  }
  double m = s / n, sd = sqrt(s2 / n - m * m);
  printf("vph n=%ld mean_uV=%.0f std_uV=%.0f sem_uV=%.0f msm_mC_start=%ld msm_mC_end=%ld\n", n, m, sd,
         sd / sqrt((double)n), t0c, rd(MSM));
  return 0;
}
