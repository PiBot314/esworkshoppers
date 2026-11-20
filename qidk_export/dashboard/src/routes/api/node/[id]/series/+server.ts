import { error, json } from '@sveltejs/kit';
import { getSeries } from '#lib/server/db.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ params, url }) => {
	const num = (k: string) => {
		const v = Number(url.searchParams.get(k));
		if (!Number.isFinite(v)) error(400, `Bad or missing "${k}"`);
		return Math.trunc(v);
	};
	return json(getSeries(params.id, num('channel'), num('source'), num('from'), num('to')));
};
