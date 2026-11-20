// The existing anomaly rule (aimet_quantization/scripts/anomaly.py), unchanged:
//   flag  <=>  |(actual - forecast) - median[hour]| > limit[hour]
// with per-IST-hour median and limit from the model's meta.json (fitted on 2025 validation data).
import type { ModelMeta } from './features.ts';

export interface Check { residual: number; median: number; limit: number; excess: number; flagged: boolean }

export function check(meta: ModelMeta, domain: number, hour: number, actual: number, forecast: number): Check {
	const residual = actual - forecast;
	const median = meta.anomaly.median_W[hour][domain];
	const limit = meta.anomaly.limit_W[hour][domain];
	const excess = Math.abs(residual - median) - limit;
	return { residual, median, limit, excess, flagged: excess > 0 };
}

/** Demo faults on the ACTUAL reading, as in anomaly.py inject() and the nz_dashboard fault buttons. */
export type FaultKind = 'dropout' | 'spike' | 'damage40';
export function inject(kind: FaultKind, actual: number, capacity: number) {
	return kind === 'dropout' ? 0 : kind === 'spike' ? actual + 0.5 * capacity : actual * 0.6;
}
