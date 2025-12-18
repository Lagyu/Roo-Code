export type ProviderLogStage = "request" | "response" | "status" | "retry" | "error"

export type ProviderLogEntry = {
	provider: string
	stage: ProviderLogStage
	requestId?: string
	model?: string
	message?: string
	data?: unknown
	error?: unknown
}

export type ProviderLogger = (entry: ProviderLogEntry & { timestamp: number }) => void

let currentLogger: ProviderLogger | undefined

export function setProviderLogger(logger?: ProviderLogger) {
	currentLogger = logger
}

export function logProviderEvent(entry: ProviderLogEntry) {
	if (!currentLogger) return

	const payload = {
		...entry,
		data: sanitize(entry.data),
		error: entry.error ? formatError(entry.error) : undefined,
		timestamp: Date.now(),
	}

	try {
		currentLogger(payload)
	} catch (err) {
		console.error("[provider-logger] failed to write log", err)
	}
}

function formatError(error: unknown) {
	if (!error) return undefined

	if (error instanceof Error) {
		const formatted: Record<string, unknown> = {
			name: error.name,
			message: sanitize(error.message),
			stack: error.stack ? sanitize(error.stack) : undefined,
		}

		if ((error as any).status) {
			formatted.status = (error as any).status
		}

		return formatted
	}

	if (typeof error === "object") {
		return sanitize(error)
	}

	return String(error)
}

// Sanitization for provider request/response logs:
// - Redact common secret-bearing keys
// - Truncate long strings (but keep array/object structure intact for debugging)
// - Prevent pathological recursion (circular references / extremely deep objects)
const MAX_STRING_LENGTH = 300
const MAX_DEPTH = 50
const SENSITIVE_KEY_SUBSTRINGS = ["authorization", "apikey", "api_key", "secret", "password"]

function isSensitiveKey(key: string): boolean {
	const lower = key.toLowerCase()

	if (SENSITIVE_KEY_SUBSTRINGS.some((sensitive) => lower.includes(sensitive))) return true

	// Treat "token" as sensitive only when it's a standalone segment (e.g. token, access_token, id-token)
	// or when the key ends with "token" (e.g. refreshToken). Do not redact token *count* fields like
	// max_output_tokens, prompt_tokens, outputTokens, or x-*-ratelimit-*-tokens.
	if (/(^|[_-])token([_-]|$)/.test(lower)) return true
	if (!lower.endsWith("tokens") && lower.endsWith("token")) return true

	return false
}

function sanitize(value: unknown, depth = 0, stack: WeakSet<object> = new WeakSet()): unknown {
	if (value === null || value === undefined) {
		return value
	}

	if (depth > MAX_DEPTH) {
		return "[truncated depth]"
	}

	if (typeof value === "string") {
		if (value.startsWith("data:")) {
			return `<data-uri length=${value.length}>`
		}
		if (value.length > MAX_STRING_LENGTH) {
			return `${value.slice(0, MAX_STRING_LENGTH)}... [truncated ${value.length - MAX_STRING_LENGTH} chars]`
		}
		return value
	}

	if (typeof value === "number" || typeof value === "boolean") {
		return value
	}

	if (value instanceof Error) {
		return formatError(value)
	}

	if (Array.isArray(value)) {
		if (stack.has(value)) {
			return "[circular]"
		}
		stack.add(value)
		try {
			return value.map((item) => sanitize(item, depth + 1, stack))
		} finally {
			stack.delete(value)
		}
	}

	if (typeof value === "object") {
		if (stack.has(value as object)) {
			return "[circular]"
		}
		stack.add(value as object)
		const out: Record<string, unknown> = {}
		try {
			for (const [key, val] of Object.entries(value as Record<string, unknown>)) {
				if (isSensitiveKey(key)) {
					out[key] = "<redacted>"
				} else {
					out[key] = sanitize(val, depth + 1, stack)
				}
			}
			return out
		} finally {
			stack.delete(value as object)
		}
	}

	return String(value)
}
