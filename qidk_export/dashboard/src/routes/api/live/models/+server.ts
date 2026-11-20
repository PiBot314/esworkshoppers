// Forecasting models the live loop can use (plan_model_switcher.md §3.4).
//   GET  /api/live/models                      -> { models, active }
//   POST /api/live/models {signal, model}       -> switch the model behind a signal (UI only; never a chat tool)
import { error, json } from '@sveltejs/kit';
import { activeModelsInfo, listModels, startLive, switchModel, type Signal } from '#lib/server/live/loop.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = () => {
	startLive();
	return json({ models: listModels(), active: activeModelsInfo() }, { headers: { 'cache-control': 'no-store' } });
};

export const POST: RequestHandler = async ({ request }) => {
	const b = (await request.json().catch(() => ({}))) as { signal?: string; model?: string };
	if (b.signal !== 'grid' && b.signal !== 'pl00') error(400, 'signal must be "grid" or "pl00"');
	if (typeof b.model !== 'string' || !/^[a-z0-9_]{1,40}$/.test(b.model)) error(400, 'bad model id');
	try {
		const r = await switchModel(b.signal as Signal, b.model);
		return json(r, { status: r.ok ? 200 : 400 });
	} catch (e) {
		return json({ ok: false, error: e instanceof Error ? e.message : String(e) }, { status: 500 });
	}
};
