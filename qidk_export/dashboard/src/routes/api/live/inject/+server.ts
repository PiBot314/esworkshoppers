// Demo fault injection (UI only; the assistant cannot call this).
// POST {signal: "grid"|"pl00", kind: "dropout"|"spike"|"damage40", steps: 1-4}
import { error, json } from '@sveltejs/kit';
import { enabled, requestInject, type Signal } from '#lib/server/live/loop.ts';
import type { FaultKind } from '#lib/server/live/detect.ts';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
	if (!enabled()) error(409, 'The live loop is off (LIVE=1 not set).');
	const b = (await request.json()) as { signal?: string; kind?: string; steps?: number };
	if (b.signal !== 'grid' && b.signal !== 'pl00') error(400, 'signal must be "grid" or "pl00"');
	if (b.kind !== 'dropout' && b.kind !== 'spike' && b.kind !== 'damage40') error(400, 'kind must be "dropout", "spike" or "damage40"');
	return json(requestInject(b.signal as Signal, b.kind as FaultKind, Number(b.steps ?? 1)));
};
