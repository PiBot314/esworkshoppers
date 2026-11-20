/** State shared between the map page and the assistant (which can open a node). */
export const ui = $state({
	selectedId: null as string | null,
	/** nodes sharing the clicked map position */
	stack: [] as string[],
	chatOpen: false
});

export interface ChatStep { name: string; args: Record<string, unknown>; href?: string }
export interface ChatTurn { role: 'user' | 'assistant'; content: string; steps?: ChatStep[]; error?: string }
/** The conversation lives here so it survives closing the assistant panel. */
export const chat = $state({ turns: [] as ChatTurn[], pending: null as string | null });
