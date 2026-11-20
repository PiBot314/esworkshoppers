import { error, json } from '@sveltejs/kit';
import { getNodeDetail } from '#lib/server/db.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ params }) => {
	const detail = getNodeDetail(params.id);
	if (!detail) error(404, 'No such node');
	return json(detail);
};
