// New anomaly events since an id (for the banner): GET /api/live/events?since=<id>
import { json } from '@sveltejs/kit';
import { enabled, liveEvents, startLive } from '#lib/server/live/loop.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url }) => {
	startLive();
	const since = Number(url.searchParams.get('since') ?? 0) || 0;
	return json({ enabled: enabled(), events: liveEvents(since).slice(-20) }, { headers: { 'cache-control': 'no-store' } });
};
