// Reverse proxy to the existing nz_dashboard (:8080) so its page, data files and NPU API are served from this
// origin under /npu-app/. Its frontend only uses relative URLs, so a <base href> on index.html is all it needs.
import type { RequestHandler } from './$types';

const NPU = (process.env.NPU_URL ?? 'http://127.0.0.1:8080').replace(/\/$/, '');
const BASE = '/npu-app/';
const PASS = ['content-type', 'cache-control', 'etag', 'last-modified'];

const proxy: RequestHandler = async ({ params, url, request }) => {
	const path = params.path ?? '';
	if (path.split('/').includes('..')) return new Response('bad path', { status: 400 });
	let res: Response;
	try {
		res = await fetch(`${NPU}/${path}${url.search}`, {
			method: request.method,
			headers: { accept: request.headers.get('accept') ?? '*/*' },
			body: request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.arrayBuffer()
		});
	} catch (e) {
		return new Response(`NPU dashboard (nz_dashboard) is not reachable at ${NPU}: ${(e as Error).message}`, {
			status: 502,
			headers: { 'content-type': 'text/plain; charset=utf-8' }
		});
	}
	const headers = new Headers();
	for (const h of PASS) {
		const v = res.headers.get(h);
		if (v) headers.set(h, v);
	}
	if (path === '' || path === 'index.html') {
		const html = (await res.text()).replace(/<head([^>]*)>/i, `<head$1><base href="${BASE}">`);
		headers.delete('etag');
		headers.delete('content-length');
		return new Response(html, { status: res.status, headers });
	}
	return new Response(res.body, { status: res.status, headers });
};

export const GET = proxy;
export const HEAD = proxy;
export const POST = proxy;
