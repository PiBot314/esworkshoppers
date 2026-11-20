import { json } from '@sveltejs/kit';
import { liveStatus, startLive } from '#lib/server/live/loop.ts';

export const GET = () => {
	startLive();
	return json(liveStatus(), { headers: { 'cache-control': 'no-store' } });
};
