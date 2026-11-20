import type { Channel } from '#lib/types.ts';

/** Vertical colours: one fixed hue per vertical, never reassigned by filtering.
 *  Markers also carry the 2-letter code, so colour is never the only cue. */
const VERTICAL_COLORS: Record<string, string> = {
	SR: '#2a78d6', WN: '#eb6834', WM: '#1baf7a', EM: '#eda100',
	AQ: '#e87ba4', SL: '#008300', CM: '#6250d6', WE: '#e34948'
};
export const OTHER_COLOR = '#52514e';
export const colorOf = (vertical: string) => VERTICAL_COLORS[vertical] ?? OTHER_COLOR;
/** Ink that stays readable on top of a vertical's fill. */
export const inkOn = (vertical: string) => (vertical === 'EM' ? '#0b0b0b' : '#ffffff');

export const LEVEL_LABEL = {
	history: 'Real history',
	few: 'A few real points',
	none: 'No real data'
} as const;

const TZ = 'Asia/Kolkata'; // campus time
const dt = (o: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat('en-GB', { timeZone: TZ, ...o });
const F_DATE = dt({ day: 'numeric', month: 'short', year: 'numeric' });
const F_DATETIME = dt({ day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
const F_DAY = dt({ day: 'numeric', month: 'short' });
const F_MONTH = dt({ month: 'short', year: '2-digit' });
const F_TIME = dt({ hour: '2-digit', minute: '2-digit' });

export const fmtDate = (ts: number) => F_DATE.format(ts * 1000);
export const fmtDateTime = (ts: number) => F_DATETIME.format(ts * 1000);
/** Axis tick label suited to the visible span (seconds). */
export function fmtTick(ts: number, span: number) {
	if (span <= 2 * 86400) return F_TIME.format(ts * 1000);
	if (span <= 120 * 86400) return F_DAY.format(ts * 1000);
	return F_MONTH.format(ts * 1000);
}

export function fmtNum(v: number | null | undefined) {
	if (v == null || !Number.isFinite(v)) return '–';
	const a = Math.abs(v);
	const digits = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : 3;
	return v.toLocaleString('en-US', { maximumFractionDigits: digits });
}
export const fmtCount = (n: number) => n.toLocaleString('en-US');

/** Convert a stored raw value to the channel's standard unit when the unit is trusted. */
export function convert(c: Channel, v: number) {
	return c.scale != null ? v * c.scale + c.offset : v;
}
export function unitOf(c: Channel) {
	// An empty scale means the stated unit is wrong or unknown, so no unit is claimed.
	const u = c.scale != null ? c.canonical_unit : null;
	return u && u !== 'n/a' ? u : '';
}
/** Short tags that tell two channels of the same quantity apart. */
export function tagsOf(c: Channel) {
	return [c.circuit, c.phase, c.qualifier !== 'raw' ? c.qualifier : null].filter(Boolean) as string[];
}
export const isChartable = (c: Channel) => c.value_kind !== 'text' && c.value_kind !== 'time';
