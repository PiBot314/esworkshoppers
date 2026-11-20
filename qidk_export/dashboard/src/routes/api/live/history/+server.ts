// Actual vs NPU forecast for the last hours, per live signal: GET /api/live/history?hours=24
import { json } from '@sveltejs/kit';
import { enabled, liveHistory, startLive } from '#lib/server/live/loop.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url }) => {
	startLive();
	const hours = Math.min(48, Math.max(1, Number(url.searchParams.get('hours') ?? 24) || 24));
	return json({ enabled: enabled(), ...liveHistory(hours) }, { headers: { 'cache-control': 'no-store' } });
};
