// Chat with a small language model that runs on this machine (llama.cpp server,
// OpenAI-compatible API). The model answers by calling the tools in tools.ts, so
// every number it gives comes from campus.sqlite.
import { json } from '@sveltejs/kit';
import { runTool, TOOL_SPECS } from '#lib/server/tools.ts';
import type { RequestHandler } from './$types';

const SLM_URL = (process.env.SLM_URL ?? 'http://127.0.0.1:8090/v1').replace(/\/$/, '');
const MAX_ROUNDS = 6;

const SYSTEM = `You are the assistant inside a dashboard of sensor data from the IIIT Hyderabad campus.
Everything you know about the campus comes from one database, which you can only read through the tools.

Rules:
- For any fact or number about nodes or their data, call a tool first. Never invent a number, a month or a reason.
- Only state what a tool result says. If the tools cannot answer, say so.
- Never say data is missing unless a tool result says so. If a result does not cover what was asked, call the tool again.
- To compare two periods, call get_stats once covering both with group_by "month", or once per period.
- Node IDs look like EM-CR-SB00-00. If you are not sure of an ID, call search_nodes.
- The database is historical, mostly the year 2025. Only about 13 nodes have a full year of real data.
- Separately, there is LIVE data: the campus API (synthetic 2026 data replayed in real time), checked every 15 minutes by forecasting models on the board's NPU. Only the live tools (live_status, recent_anomalies, explain_anomaly) see it. Say "live" when you use them.
- A live anomaly flag means a reading was far from the forecast. It is NOT proof of a fault: never call it a fault or a failure. If the tool says it was injected, say it was a demo fault.
- If the live tools show no flags in the period asked about, say plainly that there is no flag; don't describe a flag that doesn't exist (as "not a real fault" or otherwise).
- Use real ("measured") data unless the user asks for synthetic data.
- If a tool result has a unit_warning or a caveat, pass it on in one short sentence.
- Answer in at most four short plain sentences, with units. No markdown, no tables, no lists.

Which tool for which question:
- "How much / when / average / peak / highest / compare months or hours" of power, energy, voltage, solar: get_stats. Use group_by "hour" for time of day, "month" for time of year, "weekday" for day of week.
- Campus-wide electricity use: get_stats on node EM-CR-SB00-00 with value "GDPower".
- "What is this node", "what does it measure", "how much data": get_node.
- "Which nodes", "find", "list": search_nodes.
- "Which node has the most / least", ranking or comparing nodes: compare_nodes.
- Anything about water, air quality, weather, rooms or poles: search_nodes with that type first. Those nodes exist but mostly have only a few data points.
- "What data exists", "how many sensors": data_overview. Never use data_overview for power or usage questions.
- Net zero, carbon, CO2, emissions, trees, or "what if we add more solar": net_zero.
- "Show", "open", "go to" a node: show_node.
- "Now", "right now", "currently", "latest", "live", "next hour", "forecast": live_status.
- "Which model", "how accurate", "compare the models", "NPU speed": list_models.
- "Results", "better than persistence", "baseline", "skill", "is the forecast worth it", "which inputs matter": model_results.
- "Anomaly", "alert", "flag", "unusual", "anything wrong", "what happened": recent_anomalies, then explain_anomaly for the one asked about.
- "Why was it flagged", "is it a fault", "explain": explain_anomaly. Pass on its what_it_means sentences.

Useful nodes:
- EM-CR-SB00-00: the campus substation. Its value "GDPower" is the whole campus's grid power in kW.
- EM-CR-SB00-01: the diesel generators (G1Power, G2Power).
- SR-EM-KH04-00, -01, -02: air-conditioning meters on the 4th floor of Kohli Research Block.
- SL-...: solar inverters and meters (Nilgiri = NI, Palash = PL, Vindhya = VN).`;

interface Msg { role: string; content: string | null; tool_calls?: ToolCall[]; tool_call_id?: string }
interface ToolCall { id: string; type: 'function'; function: { name: string; arguments: string } }

export const GET: RequestHandler = async () => {
	try {
		const r = await fetch(`${SLM_URL}/models`, { signal: AbortSignal.timeout(1500) });
		const d = (await r.json()) as { data?: { id: string }[] };
		const id = d.data?.[0]?.id ?? 'local model';
		return json({ ok: true, model: id.split('/').pop()!.replace(/\.gguf$/i, '') });
	} catch {
		return json({ ok: false, model: null });
	}
};

/** Older llama.cpp builds refuse to stream when tools are given; remembered after the first refusal. */
let canStream = true;

/** One completion. Calls onText for each piece of answer text; returns text + tool calls. */
async function complete(messages: Msg[], onText: (s: string) => void, signal: AbortSignal, mustUseTool = false): Promise<{ text: string; calls: ToolCall[] }> {
	const req = () => fetch(`${SLM_URL}/chat/completions`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		// 250 tokens: the prompt asks for at most four short sentences; this stops a rambling answer early (≈6 tok/s on the board)
		body: JSON.stringify({ model: 'local', messages, tools: TOOL_SPECS, tool_choice: mustUseTool ? 'required' : 'auto', stream: canStream, temperature: 0.2, max_tokens: 250 }),
		signal
	});
	let res: Response;
	try {
		res = await req();
	} catch (e) {
		// a kept-alive connection that llama-server already closed: retry once on a fresh one
		if (signal.aborted || !/ECONNRESET|socket connection was closed/i.test(String(e))) throw e;
		res = await req();
	}
	if (!res.ok || !res.body) {
		const detail = (await res.text()).slice(0, 200);
		if (canStream && /tools with stream/i.test(detail)) {
			canStream = false;
			return complete(messages, onText, signal, mustUseTool);
		}
		throw new Error(`model server answered ${res.status}: ${detail}`);
	}
	if (!canStream) {
		const msg = ((await res.json()) as { choices?: { message?: { content?: string | null; tool_calls?: ToolCall[] } }[] }).choices?.[0]?.message;
		const text = msg?.content ?? '';
		if (text) onText(text);
		return { text, calls: (msg?.tool_calls ?? []).map((c) => ({ ...c, type: 'function' as const })) };
	}
	const calls: ToolCall[] = [];
	let text = '', buf = '';
	const dec = new TextDecoder();
	for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
		buf += dec.decode(chunk, { stream: true });
		let nl;
		while ((nl = buf.indexOf('\n')) >= 0) {
			const line = buf.slice(0, nl).trim();
			buf = buf.slice(nl + 1);
			if (!line.startsWith('data:')) continue;
			const payload = line.slice(5).trim();
			if (payload === '[DONE]') continue;
			let delta;
			try { delta = JSON.parse(payload).choices?.[0]?.delta; } catch { continue; }
			if (!delta) continue;
			if (typeof delta.content === 'string' && delta.content) { text += delta.content; onText(delta.content); }
			for (const tc of delta.tool_calls ?? []) {
				const c = (calls[tc.index ?? 0] ??= { id: '', type: 'function', function: { name: '', arguments: '' } });
				if (tc.id) c.id = tc.id;
				if (tc.function?.name) c.function.name += tc.function.name;
				if (tc.function?.arguments) c.function.arguments += tc.function.arguments;
			}
		}
	}
	return { text, calls: calls.filter(Boolean) };
}

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json()) as { messages?: { role: string; content: string }[]; context?: { page?: string; node?: string | null } };
	const history = (body.messages ?? [])
		.filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
		.slice(-10)
		.map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }));
	const where = body.context?.node
		? `The user is looking at node ${body.context.node} on the map. "this node" means that one.`
		: body.context?.page === '/whatif' ? 'The user is on the "Net zero" page (carbon balance and extra-solar what-if).'
		: body.context?.page === '/map' ? 'The user is on the map page with no node selected.'
		: body.context?.page === '/results' ? 'The user is on the "Results" page: model accuracy against the persistence and same-time-yesterday baselines, live shadow-mode scores, and which inputs the models rely on. Use model_results.'
		: body.context?.page === '/npu' ? 'The user is on the "NPU lab" page (replays, benchmarks and single NPU runs of the forecasting models).'
		: 'The user is on the "Live" page: live NPU forecasts and anomaly flags for campus grid draw and the Palash solar plant. Use the live tools.';
	const messages: Msg[] = [{ role: 'system', content: `${SYSTEM}\n\n${where}` }, ...history];

	const last = history[history.length - 1]?.content ?? '';
	const isQuestion = last.includes('?') || last.trim().split(/\s+/).length > 3;

	const enc = new TextEncoder();
	const stream = new ReadableStream({
		async start(ctl) {
			const send = (o: unknown) => ctl.enqueue(enc.encode(JSON.stringify(o) + '\n'));
			try {
				for (let round = 0; round < MAX_ROUNDS; round++) {
					// A real question must be looked up before it is answered; small talk need not be.
					// tool_choice "required" isn't always obeyed, so for a question the first round's text is held back,
					// and if no tool was called it is dropped and the model is asked once more, with a reminder.
					const mustUseTool = round === 0 && isQuestion;
					let held = '';
					const onText = mustUseTool ? (s: string) => { held += s; } : (s: string) => send({ type: 'text', text: s });
					let { text, calls } = await complete(messages, onText, request.signal, mustUseTool);
					if (mustUseTool && !calls.length) {
						const retry = [...messages, { role: 'system', content: 'You answered without looking anything up. Call the most fitting tool now; do not answer from memory.' }];
						({ text, calls } = await complete(retry, (s) => send({ type: 'text', text: s }), request.signal, true));
					} else if (held) send({ type: 'text', text: held });
					if (!calls.length) break;
					calls.forEach((c, i) => (c.id ||= `call_${round}_${i}`));
					messages.push({ role: 'assistant', content: text || null, tool_calls: calls });
					for (const c of calls) {
						let args: Record<string, unknown> = {};
						try { args = JSON.parse(c.function.arguments || '{}'); } catch { /* model wrote bad JSON: run with no arguments */ }
						const out = runTool(c.function.name, args);
						send({ type: 'tool', name: c.function.name, args, ui: out.ui });
						messages.push({ role: 'tool', tool_call_id: c.id, content: JSON.stringify(out.result) });
					}
					if (text) send({ type: 'text', text: '\n' });
					if (round === MAX_ROUNDS - 1) send({ type: 'text', text: 'I stopped after several lookups without reaching an answer. Try a narrower question.' });
				}
			} catch (e) {
				const msg = e instanceof Error ? e.message : String(e);
				const down = /ECONNREFUSED|Unable to connect|fetch failed|ConnectionRefused/i.test(msg);
				if (!request.signal.aborted)
					send({ type: 'error', message: down ? 'The language model is not running. Start it in another terminal with: bun run slm' : msg });
			}
			send({ type: 'done' });
			ctl.close();
		}
	});
	return new Response(stream, { headers: { 'content-type': 'application/x-ndjson', 'cache-control': 'no-store' } });
};
