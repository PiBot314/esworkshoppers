import { json } from '@sveltejs/kit';
import { runWhatIf } from '#lib/server/whatif.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url }) => {
	const q = url.searchParams;
	return json(runWhatIf({ extra_solar_pct: Number(q.get('solar')) || 0, month: q.get('month') ?? undefined, export_surplus: q.get('export') === '1' }));
};
