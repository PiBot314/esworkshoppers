// Starts the live loop once when the server starts (only if LIVE=1).
import { startLive } from '#lib/server/live/loop.ts';

export const init = async () => {
	startLive();
};

// Exit on SIGTERM/SIGINT. adapter-node's graceful shutdown closes the server, but under Bun on the board the
// process then stays alive with no server (seen twice), so start.sh never restarts it. Exit once adapter-node
// says it's done, or after 3 s at the latest.
const bye = () => process.exit(0);
process.once('sveltekit:shutdown' as never, bye);
for (const sig of ['SIGTERM', 'SIGINT'] as const) process.once(sig, () => setTimeout(bye, 3000));
