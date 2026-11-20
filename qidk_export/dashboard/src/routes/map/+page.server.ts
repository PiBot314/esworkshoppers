import { getNodes, getSources, getVerticals } from '#lib/server/db.ts';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({
	nodes: getNodes(),
	verticals: getVerticals(),
	sources: getSources()
});
