// Results tab: live accuracy (active, shadow models, baselines), offline 2025 results, per-hour limits. GET /api/live/results?hours=24
import { json } from '@sveltejs/kit';
import { enabled, resultsBundle, startLive } from '#lib/server/live/loop.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url }) => {
	startLive();
	const hours = Math.min(192, Math.max(1, Number(url.searchParams.get('hours') ?? 24) || 24));
	return json({ enabled: enabled(), hours, ...resultsBundle(hours) }, { headers: { 'cache-control': 'no-store' } });
};
