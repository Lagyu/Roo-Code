import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"

import {
	type ModelInfo,
	openAiNativeDefaultModelId,
	OpenAiNativeModelId,
	openAiNativeModels,
	OPENAI_NATIVE_DEFAULT_TEMPERATURE,
	type ReasoningEffort,
	type VerbosityLevel,
	type ReasoningEffortExtended,
	type ServiceTier,
} from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"

import { calculateApiCostOpenAI } from "../../shared/cost"

import { ApiStream, ApiStreamUsageChunk } from "../transform/stream"
import { getModelParams } from "../transform/model-params"

import { BaseProvider } from "./base-provider"
import { logProviderEvent, type ProviderLogStage } from "./provider-logger"
import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"

export type OpenAiNativeModel = ReturnType<OpenAiNativeHandler["getModel"]>

// GPT-5 specific types

// Constants for model identification
const GPT5_MODEL_PREFIX = "gpt-5"

// Default caps for max_output_tokens to avoid oversized KV allocations ("no_kv_space") in some deployments.
// Users can override via the profile's `modelMaxTokens` setting.
const OPENAI_NATIVE_DEFAULT_MAX_OUTPUT_TOKENS = 16_384
const OPENAI_NATIVE_AZURE_DEFAULT_MAX_OUTPUT_TOKENS = 8_192

// Marker for terminal background-mode failures so we don't attempt resume/poll fallbacks
function createTerminalBackgroundError(message: string): Error {
	const err = new Error(message)
	;(err as any).isTerminalBackgroundError = true
	err.name = "TerminalBackgroundError"
	return err
}
function isTerminalBackgroundError(err: any): boolean {
	return !!(err && (err as any).isTerminalBackgroundError)
}

function isAzureOpenAiBaseUrl(baseUrl?: string): boolean {
	if (!baseUrl) return false
	return baseUrl.toLowerCase().includes(".openai.azure.com")
}

function isGpt5ProModel(modelId: string): boolean {
	return modelId.startsWith("gpt-5-pro")
}

function isNoKvSpaceError(error: unknown): boolean {
	const message = error instanceof Error ? error.message : String(error)
	const lower = message.toLowerCase()
	// Different gateways/deployments surface KV-cache allocation failures with different strings.
	// Examples:
	// - "no_kv_space"
	// - "Failed to extend cache for completion ..."
	return (
		lower.includes("no_kv_space") ||
		lower.includes("failed to extend cache for completion") ||
		lower.includes("failed to extend cache")
	)
}

function getHeaderValue(headers: any, name: string): string | undefined {
	if (!headers) return undefined
	const lowerName = name.toLowerCase()

	// Standard fetch Headers
	if (typeof headers.get === "function") {
		const val = headers.get(name) ?? headers.get(lowerName)
		return typeof val === "string" && val.length > 0 ? val : undefined
	}

	// Plain object (common on SDK errors)
	const direct = headers[name]
	if (typeof direct === "string" && direct.length > 0) return direct
	const lower = headers[lowerName]
	if (typeof lower === "string" && lower.length > 0) return lower

	return undefined
}

function getRetryAfterMs(headers: any): number | undefined {
	const retryAfterMsHeader =
		getHeaderValue(headers, "retry-after-ms") || getHeaderValue(headers, "x-ms-retry-after-ms")
	if (retryAfterMsHeader) {
		const ms = Number(retryAfterMsHeader)
		if (Number.isFinite(ms) && ms > 0) return ms
	}

	const retryAfter = getHeaderValue(headers, "retry-after")
	if (!retryAfter) return undefined

	// Prefer delta seconds (most common).
	const seconds = Number(retryAfter)
	if (Number.isFinite(seconds) && seconds > 0) return seconds * 1000

	// Fallback: HTTP-date.
	const dateMs = Date.parse(retryAfter)
	if (!Number.isNaN(dateMs)) {
		const deltaMs = dateMs - Date.now()
		if (deltaMs > 0) return deltaMs
	}

	return undefined
}

function pickRateLimitHeaders(headers: any): Record<string, string> | undefined {
	const headerNames = [
		"retry-after",
		"retry-after-ms",
		"x-ms-retry-after-ms",
		"x-ms-ratelimit-limit-requests",
		"x-ms-ratelimit-remaining-requests",
		"x-ms-ratelimit-reset-requests",
		"x-ms-ratelimit-limit-tokens",
		"x-ms-ratelimit-remaining-tokens",
		"x-ms-ratelimit-reset-tokens",
		"x-ratelimit-limit-requests",
		"x-ratelimit-remaining-requests",
		"x-ratelimit-reset-requests",
		"x-ratelimit-limit-tokens",
		"x-ratelimit-remaining-tokens",
		"x-ratelimit-reset-tokens",
		"x-ratelimit-limit-input-tokens",
		"x-ratelimit-remaining-input-tokens",
		"x-ratelimit-reset-input-tokens",
		"x-ratelimit-limit-output-tokens",
		"x-ratelimit-remaining-output-tokens",
		"x-ratelimit-reset-output-tokens",
		"x-request-id",
		"openai-request-id",
		"apim-request-id",
		"x-ms-request-id",
		"x-ms-region",
	]

	const out: Record<string, string> = {}
	for (const name of headerNames) {
		const val = getHeaderValue(headers, name)
		if (val) out[name] = val
	}
	return Object.keys(out).length > 0 ? out : undefined
}

function listHeaderKeys(headers: any): string[] | undefined {
	if (!headers) return undefined
	try {
		const keys: string[] = []

		if (typeof headers.keys === "function") {
			for (const key of headers.keys()) {
				keys.push(String(key))
			}
		} else if (typeof headers.forEach === "function") {
			headers.forEach((_value: any, key: string) => {
				keys.push(String(key))
			})
		} else if (typeof headers === "object") {
			keys.push(...Object.keys(headers))
		}

		const deduped = Array.from(new Set(keys))
		deduped.sort()
		return deduped
	} catch {
		return undefined
	}
}

function headersToObject(headers: any): Record<string, string> | undefined {
	if (!headers) return undefined
	try {
		const out: Record<string, string> = {}

		// Standard fetch Headers
		if (typeof headers.forEach === "function") {
			headers.forEach((value: any, key: string) => {
				if (value === undefined || value === null) return
				out[String(key)] = String(value)
			})
		} else if (typeof headers.entries === "function") {
			for (const entry of headers.entries()) {
				const [key, value] = entry as any
				if (value === undefined || value === null) continue
				out[String(key)] = String(value)
			}
		} else if (typeof headers === "object") {
			for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
				if (value === undefined || value === null) continue
				out[String(key)] = Array.isArray(value) ? value.map((v) => String(v)).join(", ") : String(value)
			}
		}

		return Object.keys(out).length > 0 ? out : undefined
	} catch {
		return undefined
	}
}

// Provider logs sanitize/truncate long strings; chunk to preserve full raw payloads for debugging.
const PROVIDER_LOG_STRING_CHUNK_SIZE = 300
function chunkStringForProviderLog(value: string): string[] {
	if (!value) return [""]
	const out: string[] = []
	for (let i = 0; i < value.length; i += PROVIDER_LOG_STRING_CHUNK_SIZE) {
		out.push(value.slice(i, i + PROVIDER_LOG_STRING_CHUNK_SIZE))
	}
	return out
}

function extractErrorDiagnostics(err: any): Record<string, unknown> {
	const status = err?.status ?? err?.response?.status
	const requestId = err?.request_id ?? err?.requestId ?? err?.response?.request_id ?? undefined
	const headers = pickRateLimitHeaders(err?.headers ?? err?.response?.headers)
	const headerKeys = listHeaderKeys(err?.headers ?? err?.response?.headers)
	const retryAfterMs = getRetryAfterMs(err?.headers ?? err?.response?.headers)
	const code = err?.code ?? err?.error?.code
	const type = err?.type ?? err?.error?.type
	const param = err?.param ?? err?.error?.param

	return {
		name: err?.name,
		message: err?.message,
		status,
		code,
		type,
		param,
		requestId,
		headers,
		headerKeys,
		...(typeof retryAfterMs === "number" ? { retryAfterMs } : {}),
	}
}

export class OpenAiNativeHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: OpenAI
	// Resolved service tier from Responses API (actual tier used by OpenAI)
	private lastServiceTier: ServiceTier | undefined
	// Complete response output array (includes reasoning items with encrypted_content)
	private lastResponseOutput: any[] | undefined
	// Last top-level response id from Responses API (for troubleshooting)
	private lastResponseId: string | undefined
	// Last successfully stored top-level response id (for previous_response_id chaining)
	private previousResponseId: string | undefined
	// Abort controller for cancelling ongoing requests
	private abortController?: AbortController
	// Sequence number for background mode stream resumption
	private lastSequenceNumber: number | undefined
	// Track whether current request is in background mode for status chunk annotation
	private currentRequestIsBackground?: boolean
	// Cutoff sequence for filtering stale events during resume
	private resumeCutoffSequence?: number
	// Per-request tracking to prevent stale resume attempts
	private currentRequestResponseId?: string
	private currentRequestSequenceNumber?: number
	private currentRequestLogId?: string

	// Event types handled by the shared event processor to avoid duplication
	private readonly coreHandledEventTypes = new Set<string>([
		"response.text.delta",
		"response.output_text.delta",
		"response.reasoning.delta",
		"response.reasoning_text.delta",
		"response.reasoning_summary.delta",
		"response.reasoning_summary_text.delta",
		"response.refusal.delta",
		"response.output_item.added",
		"response.done",
		"response.completed",
		"response.tool_call_arguments.delta",
		"response.function_call_arguments.delta",
		"response.tool_call_arguments.done",
		"response.function_call_arguments.done",
	])

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options
		// Default to including reasoning.summary: "auto" for models that support Responses API
		// reasoning summaries unless explicitly disabled.
		if (this.options.enableResponsesReasoningSummary === undefined) {
			this.options.enableResponsesReasoningSummary = true
		}
		const apiKey = this.options.openAiNativeApiKey ?? "not-provided"
		const rawBaseUrl = this.options.openAiNativeBaseUrl
		const normalizedBaseUrl = rawBaseUrl
			? (() => {
					const normalized = rawBaseUrl.replace(/\/+$/, "")
					const hasVersion = /\/v\d+(?:\.\d+)?$/.test(normalized)
					return hasVersion ? normalized : `${normalized}/v1`
				})()
			: undefined
		this.client = new OpenAI({
			baseURL: normalizedBaseUrl,
			apiKey,
			fetch: async (input: any, init?: any) => {
				const url = typeof input === "string" ? input : input?.url
				const method = init?.method || input?.method || "GET"
				const startedAt = Date.now()

				const res = await fetch(input as any, init as any)

				// Best-effort structured logging for SDK HTTP responses to help diagnose throttling.
				// Avoid logging request headers (API keys). Only log selected response headers.
				try {
					const retryAfterMs = getRetryAfterMs((res as any)?.headers)
					const headerKeys = listHeaderKeys((res as any)?.headers)
					this.logProvider(
						"response",
						"SDK HTTP response received",
						{
							method,
							url,
							status: (res as any)?.status,
							ok: (res as any)?.ok,
							durationMs: Date.now() - startedAt,
							headers: pickRateLimitHeaders((res as any)?.headers),
							headerKeys,
							...(typeof retryAfterMs === "number" ? { retryAfterMs } : {}),
						},
						this.getModel().id,
					)
				} catch {
					// Never break requests due to logging failures.
				}

				// If we hit a 429, log the raw HTTP response (headers + body) for debugging.
				// Use Response.clone() so we don't consume the body the SDK needs.
				try {
					if ((res as any)?.status === 429) {
						let bodyText: string | undefined
						try {
							bodyText =
								typeof (res as any)?.clone === "function"
									? await (res as any).clone().text()
									: undefined
						} catch {
							bodyText = undefined
						}
						this.logProvider(
							"error",
							"SDK HTTP 429 response (raw)",
							{
								method,
								url,
								status: (res as any)?.status,
								statusText: (res as any)?.statusText,
								headers: headersToObject((res as any)?.headers),
								headerKeys: listHeaderKeys((res as any)?.headers),
								...(typeof bodyText === "string"
									? {
											bodyTextLength: bodyText.length,
											bodyTextChunks: chunkStringForProviderLog(bodyText),
										}
									: {}),
							},
							this.getModel().id,
						)
					}
				} catch {
					// Never break requests due to logging failures.
				}

				return res as any
			},
		})
	}

	private getMaxOutputTokensForRequest(model: OpenAiNativeModel): number | undefined {
		// Respect explicit user override first.
		const userMax = this.options.modelMaxTokens
		if (typeof userMax === "number" && Number.isFinite(userMax) && userMax > 0) {
			const modelCap =
				typeof model.info.maxTokens === "number" &&
				Number.isFinite(model.info.maxTokens) &&
				model.info.maxTokens > 0
					? model.info.maxTokens
					: undefined
			return modelCap ? Math.min(userMax, modelCap) : userMax
		}

		const computedMax = model.maxTokens
		if (typeof computedMax !== "number" || !Number.isFinite(computedMax) || computedMax <= 0) return undefined

		// Avoid requesting extremely large outputs by default; some deployments fail KV allocation ("no_kv_space")
		// when max_output_tokens is set near the model's upper bound.
		const cap = isAzureOpenAiBaseUrl(this.options.openAiNativeBaseUrl)
			? OPENAI_NATIVE_AZURE_DEFAULT_MAX_OUTPUT_TOKENS
			: OPENAI_NATIVE_DEFAULT_MAX_OUTPUT_TOKENS

		return computedMax > cap ? cap : computedMax
	}

	private getNoKvSpaceRetryMaxOutputTokens(currentMax: number | undefined): number[] {
		const candidates = isAzureOpenAiBaseUrl(this.options.openAiNativeBaseUrl)
			? [4096, 2048, 1024, 512]
			: [8192, 4096, 2048, 1024, 512]
		return candidates.filter((c) => typeof currentMax !== "number" || c < currentMax)
	}

	private buildResponsesUrl(path: string): string {
		const rawBase = this.options.openAiNativeBaseUrl || "https://api.openai.com"
		// Normalize base by trimming trailing slashes
		const normalizedBase = rawBase.replace(/\/+$/, "")
		// If the base already ends with a version segment (e.g. /v1), do not append another
		const hasVersion = /\/v\d+(?:\.\d+)?$/.test(normalizedBase)
		const baseWithVersion = hasVersion ? normalizedBase : `${normalizedBase}/v1`
		const normalizedPath = path.startsWith("/") ? path : `/${path}`
		return `${baseWithVersion}${normalizedPath}`
	}

	private createRequestLogId(metadata?: ApiHandlerCreateMessageMetadata): string {
		const prefix = metadata?.taskId ? `task-${metadata.taskId}` : "req"
		return `${prefix}-${Date.now().toString(36)}`
	}

	private logProvider(stage: ProviderLogStage, message: string, data?: unknown, modelId?: string) {
		logProviderEvent({
			provider: "openai-native",
			requestId: this.currentRequestLogId,
			model: modelId ?? this.getModel().id,
			stage,
			message,
			data,
		})
	}

	private normalizeUsage(usage: any, model: OpenAiNativeModel): ApiStreamUsageChunk | undefined {
		if (!usage) return undefined

		// Prefer detailed shapes when available (Responses API)
		const inputDetails = usage.input_tokens_details ?? usage.prompt_tokens_details

		// Extract cache information from details with better readability
		const hasCachedTokens = typeof inputDetails?.cached_tokens === "number"
		const hasCacheMissTokens = typeof inputDetails?.cache_miss_tokens === "number"
		const cachedFromDetails = hasCachedTokens ? inputDetails.cached_tokens : 0
		const missFromDetails = hasCacheMissTokens ? inputDetails.cache_miss_tokens : 0

		// If total input tokens are missing but we have details, derive from them
		let totalInputTokens = usage.input_tokens ?? usage.prompt_tokens ?? 0
		if (totalInputTokens === 0 && inputDetails && (cachedFromDetails > 0 || missFromDetails > 0)) {
			totalInputTokens = cachedFromDetails + missFromDetails
		}

		const totalOutputTokens = usage.output_tokens ?? usage.completion_tokens ?? 0

		// Note: missFromDetails is NOT used as fallback for cache writes
		// Cache miss tokens represent tokens that weren't found in cache (part of input)
		// Cache write tokens represent tokens being written to cache for future use
		const cacheWriteTokens = usage.cache_creation_input_tokens ?? usage.cache_write_tokens ?? 0

		const cacheReadTokens =
			usage.cache_read_input_tokens ?? usage.cache_read_tokens ?? usage.cached_tokens ?? cachedFromDetails ?? 0

		// Resolve effective tier: prefer actual tier from response; otherwise requested tier
		const effectiveTier =
			this.lastServiceTier || (this.options.openAiNativeServiceTier as ServiceTier | undefined) || undefined
		const effectiveInfo = this.applyServiceTierPricing(model.info, effectiveTier)

		// Pass total input tokens directly to calculateApiCostOpenAI
		// The function handles subtracting both cache reads and writes internally
		const { totalCost } = calculateApiCostOpenAI(
			effectiveInfo,
			totalInputTokens,
			totalOutputTokens,
			cacheWriteTokens,
			cacheReadTokens,
		)

		const reasoningTokens =
			typeof usage.output_tokens_details?.reasoning_tokens === "number"
				? usage.output_tokens_details.reasoning_tokens
				: undefined

		const out: ApiStreamUsageChunk = {
			type: "usage",
			// Keep inputTokens as TOTAL input to preserve correct context length
			inputTokens: totalInputTokens,
			outputTokens: totalOutputTokens,
			cacheWriteTokens,
			cacheReadTokens,
			...(typeof reasoningTokens === "number" ? { reasoningTokens } : {}),
			totalCost,
		}
		return out
	}

	override async *createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream {
		const model = this.getModel()

		// Use Responses API for ALL models
		yield* this.handleResponsesApiMessage(model, systemPrompt, messages, metadata)
	}

	private async *handleResponsesApiMessage(
		model: OpenAiNativeModel,
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream {
		// Reset resolved tier for this request; will be set from response if present
		this.lastServiceTier = undefined
		// Reset output array to capture current response output items
		this.lastResponseOutput = undefined
		// Reset last response id for this request
		this.lastResponseId = undefined

		// Use Responses API for ALL models
		const { verbosity, reasoning } = this.getModel()

		const willUseBackgroundMode =
			this.options.openAiNativeBackgroundMode === true || model.info.backgroundMode === true

		// Resolve reasoning effort for models that support it
		const reasoningEffort = this.getReasoningEffort(model)

		const canUsePreviousResponseId =
			willUseBackgroundMode && !!this.previousResponseId && metadata?.suppressPreviousResponseId !== true

		// In stored/background mode, do not replay encrypted reasoning items into `input`.
		// When chaining with previous_response_id, only send the incremental user turn.
		const messagesForRequest = canUsePreviousResponseId
			? this.getIncrementalMessagesForChainedRequest(messages)
			: willUseBackgroundMode
				? this.stripReasoningItems(messages)
				: messages

		// Format conversation for the Responses API using structured items
		const formattedInput = this.formatFullConversation(systemPrompt, messagesForRequest)

		// Build request body
		const requestBody = this.buildRequestBody(
			model,
			formattedInput,
			systemPrompt,
			verbosity,
			reasoningEffort,
			metadata,
			{
				previousResponseId: canUsePreviousResponseId ? this.previousResponseId : undefined,
				omitEncryptedReasoningInclude: canUsePreviousResponseId,
			},
		)

		const initialMax =
			typeof requestBody?.max_output_tokens === "number" && Number.isFinite(requestBody.max_output_tokens)
				? requestBody.max_output_tokens
				: undefined

		const retryMaxTokens = this.getNoKvSpaceRetryMaxOutputTokens(initialMax)
		const attemptBodies = [
			requestBody,
			...retryMaxTokens.map((max) => ({
				...requestBody,
				max_output_tokens: max,
			})),
		]

		for (let attemptIndex = 0; attemptIndex < attemptBodies.length; attemptIndex++) {
			const body = attemptBodies[attemptIndex]
			const currentMax =
				typeof body?.max_output_tokens === "number" && Number.isFinite(body.max_output_tokens)
					? body.max_output_tokens
					: undefined
			let emittedContent = false
			try {
				for await (const chunk of this.executeRequest(body, model, metadata, systemPrompt, messages)) {
					if (
						chunk.type === "text" ||
						chunk.type === "reasoning" ||
						chunk.type === "tool_call" ||
						chunk.type === "usage"
					) {
						emittedContent = true
					}
					yield chunk
				}

				// Track last successfully stored response id for chaining.
				if (body?.store === true && this.lastResponseId) {
					this.previousResponseId = this.lastResponseId
				}
				return
			} catch (error) {
				const isLastAttempt = attemptIndex === attemptBodies.length - 1
				if (!isLastAttempt && isNoKvSpaceError(error) && !emittedContent) {
					const nextMax = retryMaxTokens[attemptIndex]
					this.logProvider(
						"status",
						"Retrying after KV cache allocation failure with reduced max_output_tokens",
						{ from: currentMax, to: nextMax },
						model.id,
					)
					continue
				}
				throw error
			}
		}
	}

	private stripReasoningItems(messages: Anthropic.Messages.MessageParam[]): Anthropic.Messages.MessageParam[] {
		return messages.filter((m) => (m as any)?.type !== "reasoning")
	}

	private getIncrementalMessagesForChainedRequest(
		messages: Anthropic.Messages.MessageParam[],
	): Anthropic.Messages.MessageParam[] {
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i] as any
			if (msg && typeof msg === "object" && msg.role === "user") {
				return [msg]
			}
		}
		return []
	}

	private buildRequestBody(
		model: OpenAiNativeModel,
		formattedInput: any,
		systemPrompt: string,
		verbosity: any,
		reasoningEffort: ReasoningEffortExtended | undefined,
		metadata?: ApiHandlerCreateMessageMetadata,
		options?: { previousResponseId?: string; omitEncryptedReasoningInclude?: boolean },
	): any {
		// Ensure all properties are in the required array for OpenAI's strict mode
		// This recursively processes nested objects and array items
		const ensureAllRequired = (schema: any): any => {
			if (!schema || typeof schema !== "object" || schema.type !== "object") {
				return schema
			}

			const result = { ...schema }

			if (result.properties) {
				const allKeys = Object.keys(result.properties)
				result.required = allKeys

				// Recursively process nested objects
				const newProps = { ...result.properties }
				for (const key of allKeys) {
					const prop = newProps[key]
					if (prop.type === "object") {
						newProps[key] = ensureAllRequired(prop)
					} else if (prop.type === "array" && prop.items?.type === "object") {
						newProps[key] = {
							...prop,
							items: ensureAllRequired(prop.items),
						}
					}
				}
				result.properties = newProps
			}

			return result
		}

		// Build a request body for the OpenAI Responses API.
		// Ensure we explicitly pass max_output_tokens based on Roo's reserved model response calculation
		// so requests do not default to very large limits (e.g., 120k).
		type ResponsesTool =
			| {
					type: "function"
					name: string
					description?: string
					parameters?: any
					strict?: boolean
			  }
			| {
					type: "web_search_preview"
			  }

		interface ResponsesRequestBody {
			model: string
			input: Array<{ role: "user" | "assistant"; content: any[] } | { type: string; content: string }>
			stream: boolean
			reasoning?: { effort?: ReasoningEffortExtended; summary?: "auto" }
			text?: { verbosity: VerbosityLevel }
			temperature?: number
			max_output_tokens?: number
			store?: boolean
			instructions?: string
			service_tier?: ServiceTier
			include?: string[]
			previous_response_id?: string
			/** Prompt cache retention policy: "in_memory" (default) or "24h" for extended caching */
			prompt_cache_retention?: "in_memory" | "24h"
			tools?: ResponsesTool[]
			tool_choice?: any
			parallel_tool_calls?: boolean
			background?: boolean
		}

		// Validate requested tier against model support; if not supported, omit.
		const requestedTier = (this.options.openAiNativeServiceTier as ServiceTier | undefined) || undefined
		const allowedTierNames = new Set(model.info.tiers?.map((t) => t.name).filter(Boolean) || [])

		// Decide whether to enable extended prompt cache retention for this request
		const promptCacheRetention = this.getPromptCacheRetention(model)
		const maxOutputTokens = this.getMaxOutputTokensForRequest(model)
		const tools: ResponsesTool[] = []

		if (metadata?.tools) {
			tools.push(
				...metadata.tools
					.filter((tool) => tool.type === "function")
					.map((tool) => ({
						type: "function",
						name: tool.function.name,
						description: tool.function.description,
						parameters: ensureAllRequired(tool.function.parameters),
						strict: true,
					})),
			)
		}

		if (this.options.openAiNativeWebSearchPreview === true) {
			tools.push({ type: "web_search_preview" })
		}

		const body: ResponsesRequestBody = {
			model: model.id,
			input: formattedInput,
			stream: true,
			// Always use stateless operation with encrypted reasoning
			store: false,
			// Always include instructions (system prompt) for Responses API.
			// Unlike Chat Completions, system/developer roles in input have no special semantics here.
			// The official way to set system behavior is the top-level `instructions` field.
			instructions: systemPrompt,
			// Chain from the last stored response id when available.
			...(options?.previousResponseId ? { previous_response_id: options.previousResponseId } : {}),
			// Only include encrypted reasoning content when reasoning effort is set
			...(!options?.omitEncryptedReasoningInclude && reasoningEffort
				? { include: ["reasoning.encrypted_content"] }
				: {}),
			...(reasoningEffort
				? {
						reasoning: {
							...(reasoningEffort ? { effort: reasoningEffort } : {}),
							...(this.options.enableResponsesReasoningSummary ? { summary: "auto" as const } : {}),
						},
					}
				: {}),
			// Only include temperature if the model supports it
			...(model.info.supportsTemperature !== false && {
				temperature: this.options.modelTemperature ?? OPENAI_NATIVE_DEFAULT_TEMPERATURE,
			}),
			// Explicitly include the calculated max output tokens.
			// Use the per-request reserved output computed by Roo (params.maxTokens from getModelParams).
			...(maxOutputTokens ? { max_output_tokens: maxOutputTokens } : {}),
			// Include tier when selected and supported by the model, or when explicitly "default"
			...(requestedTier &&
				(requestedTier === "default" || allowedTierNames.has(requestedTier)) && {
					service_tier: requestedTier,
				}),
			// Enable extended prompt cache retention for models that support it.
			// This uses the OpenAI Responses API `prompt_cache_retention` parameter.
			...(promptCacheRetention ? { prompt_cache_retention: promptCacheRetention } : {}),
			...(tools.length > 0 ? { tools } : {}),
			...(metadata?.tool_choice && { tool_choice: metadata.tool_choice }),
		}

		// For native tool protocol, control parallel tool calls based on the metadata flag.
		// When parallelToolCalls is true, allow parallel tool calls (OpenAI's parallel_tool_calls=true).
		// When false (default), explicitly disable parallel tool calls (false).
		// For XML or when protocol is unset, omit the field entirely so the API default applies.
		if (metadata?.toolProtocol === "native") {
			body.parallel_tool_calls = metadata.parallelToolCalls ?? false
		}

		// Include text.verbosity only when the model explicitly supports it
		if (model.info.supportsVerbosity === true) {
			body.text = { verbosity: (verbosity || "medium") as VerbosityLevel }
		}

		// Enable background mode when either explicitly opted in or required by model metadata
		if (this.options.openAiNativeBackgroundMode === true || model.info.backgroundMode === true) {
			// Azure OpenAI gateways commonly enforce aggressive idle timeouts on long-lived SSE streams.
			// For long-running background requests (notably gpt-5-pro), prefer background + polling (stream:false)
			// to avoid repeated 408s during resume attempts.
			const disableBackgroundStreaming =
				model.info.backgroundMode === true && isAzureOpenAiBaseUrl(this.options.openAiNativeBaseUrl)

			body.background = true
			body.stream = disableBackgroundStreaming ? false : true
			body.store = true
		}

		return body
	}

	private async *executeRequest(
		requestBody: any,
		model: OpenAiNativeModel,
		metadata?: ApiHandlerCreateMessageMetadata,
		systemPrompt?: string,
		messages?: Anthropic.Messages.MessageParam[],
	): ApiStream {
		// Create AbortController for cancellation
		this.abortController = new AbortController()

		// Track request for logging
		this.currentRequestLogId = this.createRequestLogId(metadata)

		// Annotate if this request uses background mode (used for status chunks)
		this.currentRequestIsBackground = !!requestBody?.background
		// Reset per-request tracking to prevent stale values from previous requests
		this.currentRequestResponseId = undefined
		this.currentRequestSequenceNumber = undefined

		this.logProvider(
			"request",
			"Starting OpenAI Responses API request",
			{
				baseUrl: this.options.openAiNativeBaseUrl || "https://api.openai.com",
				background: this.currentRequestIsBackground,
				store: requestBody?.store,
				body: requestBody,
			},
			model.id,
		)

		const canAttemptResume = () =>
			this.currentRequestIsBackground &&
			(this.options.openAiNativeBackgroundAutoResume ?? true) &&
			!!this.currentRequestResponseId &&
			typeof this.currentRequestSequenceNumber === "number"

		try {
			// Use the official SDK
			const result = await (this.client as any).responses.create(requestBody, {
				signal: this.abortController.signal,
			})

			const isAsyncIterable = (value: any): value is AsyncIterable<any> =>
				!!value && typeof value[Symbol.asyncIterator] === "function"

			// Non-streaming background requests (stream:false) return a response object; poll until terminal.
			if (!isAsyncIterable(result)) {
				if (requestBody?.stream === false) {
					yield* this.handleNonStreamingApiResponse(result, requestBody, model)
					return
				}

				this.logProvider(
					"error",
					"SDK returned non-iterable stream; falling back to SSE",
					{ responseType: typeof result },
					model.id,
				)
				throw new Error(
					"OpenAI SDK did not return an AsyncIterable for Responses API streaming. Falling back to SSE.",
				)
			}

			const stream = result

			try {
				for await (const event of stream) {
					// Check if request was aborted
					if (this.abortController?.signal.aborted) {
						break
					}

					for await (const outChunk of this.processEvent(event, model)) {
						yield outChunk
					}
				}
			} catch (iterErr) {
				this.logProvider(
					"error",
					"Streaming iterator failed",
					{ error: iterErr, diagnostics: extractErrorDiagnostics(iterErr) },
					model.id,
				)
				// If terminal failure, propagate and do not attempt resume/poll
				if (isTerminalBackgroundError(iterErr)) {
					throw iterErr
				}
				// Stream dropped mid-flight; attempt resume for background requests
				if (canAttemptResume()) {
					for await (const chunk of this.attemptResumeOrPoll(
						this.currentRequestResponseId!,
						this.currentRequestSequenceNumber!,
						model,
					)) {
						yield chunk
					}
					return
				}
				throw iterErr
			}
		} catch (sdkErr: any) {
			// Terminal background failures should not trigger fallback logic (and are not SDK errors).
			if (isTerminalBackgroundError(sdkErr)) {
				this.logProvider(
					"error",
					"Background request failed (terminal)",
					{ error: sdkErr, diagnostics: extractErrorDiagnostics(sdkErr) },
					model.id,
				)
				throw sdkErr
			}
			this.logProvider(
				"error",
				"SDK responses.create failed; trying SSE fallback",
				{ error: sdkErr, diagnostics: extractErrorDiagnostics(sdkErr) },
				model.id,
			)
			// For errors, fallback to manual SSE via fetch
			try {
				yield* this.makeResponsesApiRequest(requestBody, model, metadata, systemPrompt, messages)
			} catch (fallbackErr) {
				this.logProvider("error", "SSE fallback failed", { error: fallbackErr }, model.id)
				// If SSE fallback fails mid-stream and we can resume, try that
				if (isTerminalBackgroundError(fallbackErr)) {
					throw fallbackErr
				}
				if (canAttemptResume()) {
					for await (const chunk of this.attemptResumeOrPoll(
						this.currentRequestResponseId!,
						this.currentRequestSequenceNumber!,
						model,
					)) {
						yield chunk
					}
					return
				}
				throw fallbackErr
			}
		} finally {
			this.abortController = undefined
			// Always clear background flag at end of request lifecycle
			this.currentRequestIsBackground = undefined
			this.logProvider(
				"response",
				"Responses API request finished",
				{ responseId: this.lastResponseId },
				model.id,
			)
			this.currentRequestLogId = undefined
		}
	}

	private formatFullConversation(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): any {
		// Format the entire conversation history for the Responses API using structured format
		// The Responses API (like Realtime API) accepts a list of items, which can be messages, function calls, or function call outputs.
		const formattedInput: any[] = []

		// When mixing items like `reasoning` / `function_call` with messages, the Responses API expects
		// messages in the explicit item form (`type: "message"`). Some gateways (notably Azure) will
		// reject standalone `reasoning` items if the adjacent messages are sent in the legacy
		// `{ role, content }` shape.
		const pushMessage = (role: "user" | "assistant", content: any[]) => {
			formattedInput.push({ type: "message", role, content })
		}

		// Do NOT embed the system prompt as a developer message in the Responses API input.
		// The Responses API treats roles as free-form; use the top-level `instructions` field instead.

		// Process each message
		for (const message of messages) {
			// Check if this is a reasoning item (already formatted in API history)
			if ((message as any).type === "reasoning") {
				// Pass through reasoning items as-is
				formattedInput.push(message)
				continue
			}

			if (message.role === "user") {
				const content: any[] = []
				const toolResults: any[] = []

				if (typeof message.content === "string") {
					content.push({ type: "input_text", text: message.content })
				} else if (Array.isArray(message.content)) {
					for (const block of message.content) {
						if (block.type === "text") {
							content.push({ type: "input_text", text: block.text })
						} else if (block.type === "image") {
							const image = block as Anthropic.Messages.ImageBlockParam
							const imageUrl = `data:${image.source.media_type};base64,${image.source.data}`
							content.push({ type: "input_image", image_url: imageUrl })
						} else if (block.type === "tool_result") {
							// Map Anthropic tool_result to Responses API function_call_output item
							const result =
								typeof block.content === "string"
									? block.content
									: block.content?.map((c) => (c.type === "text" ? c.text : "")).join("") || ""
							toolResults.push({
								type: "function_call_output",
								call_id: block.tool_use_id,
								output: result,
							})
						}
					}
				}

				// Add user message first
				if (content.length > 0) {
					pushMessage("user", content)
				}

				// Add tool results as separate items
				if (toolResults.length > 0) {
					formattedInput.push(...toolResults)
				}
			} else if (message.role === "assistant") {
				const content: any[] = []
				const toolCalls: any[] = []

				if (typeof message.content === "string") {
					content.push({ type: "output_text", text: message.content })
				} else if (Array.isArray(message.content)) {
					for (const block of message.content) {
						if (block.type === "text") {
							content.push({ type: "output_text", text: block.text })
						} else if (block.type === "tool_use") {
							// Map Anthropic tool_use to Responses API function_call item
							toolCalls.push({
								type: "function_call",
								call_id: block.id,
								name: block.name,
								arguments: JSON.stringify(block.input),
							})
						}
					}
				}

				// Add assistant message. If this assistant turn only contains tool calls (no text),
				// still add an empty assistant message to keep Responses API input valid when
				// preceding reasoning items are present.
				if (content.length > 0) {
					pushMessage("assistant", content)
				} else if (toolCalls.length > 0) {
					pushMessage("assistant", [{ type: "output_text", text: "" }])
				}

				// Add tool calls as separate items
				if (toolCalls.length > 0) {
					formattedInput.push(...toolCalls)
				}
			}
		}

		// Ensure each `reasoning` item is followed by its required subsequent message item.
		// Some gateways (notably Azure) will hard-fail the request if a reasoning item is not
		// immediately followed by an assistant message item, even if the rest of the history is valid.
		const ensureReasoningPaired = (items: any[]): any[] => {
			const out: any[] = []

			const isAssistantMessageItem = (item: any): boolean => {
				if (!item || typeof item !== "object") return false
				// Explicit item form
				if (item.type === "message" && item.role === "assistant" && Array.isArray(item.content)) return true
				// Legacy message form (defensive; should not happen in our formatter)
				if (!item.type && item.role === "assistant" && Array.isArray(item.content)) return true
				return false
			}

			for (let i = 0; i < items.length; i++) {
				const item = items[i]
				out.push(item)

				if (item?.type === "reasoning") {
					const next = items[i + 1]
					if (!isAssistantMessageItem(next)) {
						out.push({
							type: "message",
							role: "assistant",
							content: [{ type: "output_text", text: "" }],
						})
					}
				}
			}

			return out
		}

		return ensureReasoningPaired(formattedInput)
	}

	private async *makeResponsesApiRequest(
		requestBody: any,
		model: OpenAiNativeModel,
		metadata?: ApiHandlerCreateMessageMetadata,
		systemPrompt?: string,
		messages?: Anthropic.Messages.MessageParam[],
	): ApiStream {
		const apiKey = this.options.openAiNativeApiKey ?? "not-provided"
		const url = this.buildResponsesUrl("responses")
		const isStreamingRequest = requestBody?.stream !== false

		// Create AbortController for cancellation
		this.abortController = new AbortController()

		try {
			this.logProvider(
				"request",
				"POST /v1/responses (fetch fallback)",
				{
					url,
					background: requestBody?.background,
					stream: isStreamingRequest,
					body: requestBody,
				},
				model.id,
			)

			const response = await fetch(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
					Accept: isStreamingRequest ? "text/event-stream" : "application/json",
				},
				body: JSON.stringify(requestBody),
				signal: this.abortController.signal,
			})

			const responseHeaders = pickRateLimitHeaders(response.headers)
			const headerKeys = listHeaderKeys(response.headers)
			const retryAfterMs = getRetryAfterMs(response.headers)

			this.logProvider(
				"response",
				"Responses API fetch completed",
				{
					status: response.status,
					ok: response.ok,
					headers: responseHeaders,
					headerKeys,
					...(typeof retryAfterMs === "number" ? { retryAfterMs } : {}),
				},
				model.id,
			)

			if (!response.ok) {
				const errorText = await response.text()
				const statusLine = `Responses API HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`
				const errorMessage = [
					statusLine,
					responseHeaders ? `headers: ${JSON.stringify(responseHeaders)}` : "",
					`body: ${errorText}`,
				]
					.filter(Boolean)
					.join("\n")

				this.logProvider(
					"error",
					"Responses API returned error response",
					{
						method: "POST",
						url,
						status: response.status,
						details: errorText,
						headers: responseHeaders,
						allHeaders: headersToObject(response.headers),
						bodyTextLength: errorText.length,
						bodyTextChunks: chunkStringForProviderLog(errorText),
						headerKeys,
						...(typeof retryAfterMs === "number" ? { retryAfterMs } : {}),
					},
					model.id,
				)

				throw new Error(errorMessage)
			}

			if (isStreamingRequest) {
				if (!response.body) {
					this.logProvider(
						"error",
						"Responses API returned no response body",
						{ status: response.status },
						model.id,
					)
					throw new Error("Responses API error: No response body")
				}

				// Handle streaming response
				yield* this.handleStreamResponse(response.body, model)
				return
			}

			let json: any
			try {
				json = await response.json()
			} catch {
				this.logProvider(
					"error",
					"Responses API returned non-JSON response for non-streaming request",
					{ status: response.status },
					model.id,
				)
				throw new Error("Responses API error: Non-JSON response for non-streaming request")
			}

			yield* this.handleNonStreamingApiResponse(json, requestBody, model)
		} catch (error) {
			this.logProvider("error", "Failed to connect to Responses API", { error }, model.id)
			if (error instanceof Error) {
				// Re-throw with the original error message if it's already formatted
				if (error.message.includes("Responses API")) {
					throw error
				}
				// Otherwise, wrap it with context
				throw new Error(`Failed to connect to Responses API: ${error.message}`)
			}
			// Handle non-Error objects
			throw new Error("Unexpected error connecting to Responses API")
		} finally {
			this.abortController = undefined
		}
	}

	private async *handleNonStreamingApiResponse(raw: any, requestBody: any, model: OpenAiNativeModel): ApiStream {
		const resp = raw?.response ?? raw

		// Capture resolved service tier if present
		if (resp?.service_tier) {
			this.lastServiceTier = resp.service_tier as ServiceTier
		}
		// Capture complete output array (includes reasoning items with encrypted_content)
		if (Array.isArray(resp?.output)) {
			this.lastResponseOutput = resp.output
		}
		// Capture top-level response id
		if (typeof resp?.id === "string") {
			this.lastResponseId = resp.id as string
			this.currentRequestResponseId = resp.id as string
		}

		const status: string | undefined = resp?.status

		// Background + polling: non-streaming create returns quickly with id/status, then we poll until terminal.
		if (requestBody?.background) {
			const responseId = (resp?.id ?? this.currentRequestResponseId) as string | undefined
			if (!responseId) {
				throw new Error("Background response missing id; cannot poll")
			}

			const knownStatus =
				status === "queued" ||
				status === "in_progress" ||
				status === "completed" ||
				status === "failed" ||
				status === "canceled"
					? (status as any)
					: undefined

			if (knownStatus) {
				yield {
					type: "status",
					mode: "background",
					status: knownStatus,
					responseId,
				}

				if (knownStatus === "completed") {
					yield* this.emitFinalResponseOutput(resp, raw, model)
					return
				}
				if (knownStatus === "failed" || knownStatus === "canceled") {
					const detail: string | undefined = resp?.error?.message ?? raw?.error?.message
					throw createTerminalBackgroundError(
						detail ? `Response ${knownStatus}: ${detail}` : `Response ${knownStatus}: ${responseId}`,
					)
				}
			}

			yield* this.pollBackgroundResponse(responseId, model, { lastEmittedStatus: knownStatus })
			return
		}

		// Non-background non-streaming: treat as completed response and synthesize output/usage.
		yield* this.emitFinalResponseOutput(resp, raw, model)
	}

	private async *emitFinalResponseOutput(resp: any, raw: any, model: OpenAiNativeModel): ApiStream {
		const output = resp?.output ?? raw?.output
		let emittedAssistantContent = false
		let emittedTextChunks = 0
		let emittedTextChars = 0
		let emittedToolCalls = 0
		const emittedToolCallNames: string[] = []
		let emittedReasoningChars = 0
		let usedOutputTextFallback = false
		if (Array.isArray(output)) {
			// Ensure lastResponseOutput is populated for continuity features (encrypted reasoning).
			this.lastResponseOutput = output
			for await (const chunk of this.emitOutputItems(output, model)) {
				if (chunk.type === "text") {
					emittedAssistantContent = true
					emittedTextChunks += 1
					emittedTextChars += chunk.text.length
				} else if (chunk.type === "tool_call") {
					emittedAssistantContent = true
					emittedToolCalls += 1
					if (typeof chunk.name === "string" && chunk.name.length > 0) {
						emittedToolCallNames.push(chunk.name)
					}
				} else if (chunk.type === "reasoning") {
					emittedReasoningChars += chunk.text.length
				}
				yield chunk
			}
		}

		// Some gateways return only `output_text` on the terminal response (no `output` array).
		// If we emit no assistant content (text/tool_call), fall back to the consolidated output_text.
		if (!emittedAssistantContent) {
			const outputText = resp?.output_text ?? raw?.output_text
			if (typeof outputText === "string" && outputText.trim().length > 0) {
				yield { type: "text", text: outputText }
				emittedAssistantContent = true
				usedOutputTextFallback = true
				emittedTextChunks += 1
				emittedTextChars += outputText.length
			}
		}

		if (!emittedAssistantContent && (resp?.status === "completed" || raw?.status === "completed")) {
			this.logProvider(
				"error",
				"Completed response contained no assistant output",
				{
					responseId: resp?.id ?? raw?.id ?? this.currentRequestResponseId,
					hasOutputArray: Array.isArray(output),
					outputArrayLength: Array.isArray(output) ? output.length : undefined,
					hasOutputText: typeof (resp?.output_text ?? raw?.output_text) === "string",
					outputTextLength:
						typeof (resp?.output_text ?? raw?.output_text) === "string"
							? String(resp?.output_text ?? raw?.output_text).length
							: undefined,
					keys: Object.keys(resp ?? raw ?? {}),
				},
				model.id,
			)
		}

		// Final per-response emission summary (helps debug empty assistant output / retry loops).
		try {
			const outputText = resp?.output_text ?? raw?.output_text
			const usage = resp?.usage ?? raw?.usage
			const usageKeys = usage && typeof usage === "object" ? Object.keys(usage) : undefined
			const outputItemTypes = Array.isArray(output)
				? Array.from(
						new Set(
							output
								.map((o: any) => (o && typeof o === "object" ? (o as any).type : undefined))
								.filter((t: any) => typeof t === "string"),
						),
					)
				: undefined
			const toolCallNames = Array.from(new Set(emittedToolCallNames)).slice(0, 10)
			this.logProvider(
				"response",
				"Final response output summary",
				{
					responseId: resp?.id ?? raw?.id ?? this.currentRequestResponseId ?? this.lastResponseId,
					status: resp?.status ?? raw?.status,
					outputArrayLength: Array.isArray(output) ? output.length : undefined,
					outputItemTypes,
					outputTextLength: typeof outputText === "string" ? outputText.length : undefined,
					usageKeys,
					emittedAssistantContent,
					emittedTextChunks,
					emittedTextChars,
					emittedToolCalls,
					toolCallNames,
					emittedReasoningChars,
					usedOutputTextFallback,
				},
				model.id,
			)
		} catch {
			// Never break streaming due to logging failures.
		}

		const usage = resp?.usage ?? raw?.usage
		const usageData = this.normalizeUsage(usage, model)
		if (usageData) {
			yield usageData
		}
	}

	private async *emitOutputItems(output: any[], model: OpenAiNativeModel): ApiStream {
		for (const outputItem of output) {
			if (!outputItem || typeof outputItem !== "object") continue

			// Some deployments return output items directly as { type: "output_text", text: "..." }.
			if (outputItem.type === "output_text" && typeof (outputItem as any).text === "string") {
				yield { type: "text", text: (outputItem as any).text }
				continue
			}

			if (outputItem.type === "text" && Array.isArray(outputItem.content)) {
				for (const content of outputItem.content) {
					if (content?.type === "text" && typeof content.text === "string") {
						yield { type: "text", text: content.text }
					}
				}
				continue
			}

			if (outputItem.type === "message" && Array.isArray(outputItem.content)) {
				for (const content of outputItem.content) {
					if (
						(content?.type === "output_text" || content?.type === "text") &&
						typeof content.text === "string"
					) {
						yield { type: "text", text: content.text }
					}
				}
				continue
			}

			if (outputItem.type === "reasoning" && Array.isArray(outputItem.summary)) {
				for (const summary of outputItem.summary) {
					if (summary?.type === "summary_text" && typeof summary.text === "string") {
						yield { type: "reasoning", text: summary.text }
					}
				}
				continue
			}

			// Non-streaming tool/function calls are delivered as complete output items.
			if (outputItem.type === "function_call" || outputItem.type === "tool_call") {
				const callId = outputItem.call_id || outputItem.tool_call_id || outputItem.id
				if (!callId) continue

				const args = outputItem.arguments || outputItem.function?.arguments || outputItem.function_arguments
				yield {
					type: "tool_call",
					id: callId,
					name: outputItem.name || outputItem.function?.name || outputItem.function_name || "",
					arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
				}
				continue
			}
		}
	}

	private getBackgroundPollMaxMinutes(model: OpenAiNativeModel): number {
		const configured = this.options.openAiNativeBackgroundPollMaxMinutes
		if (typeof configured === "number" && Number.isFinite(configured) && configured > 0) {
			return configured
		}
		// gpt-5-pro is expected to take much longer than typical models.
		if (isGpt5ProModel(model.id)) return 60
		return 20
	}

	private async *pollBackgroundResponse(
		responseId: string,
		model: OpenAiNativeModel,
		options?: { lastEmittedStatus?: "queued" | "in_progress" | "completed" | "failed" | "canceled" },
	): ApiStream {
		const apiKey = this.options.openAiNativeApiKey ?? "not-provided"
		const pollIntervalMs = this.options.openAiNativeBackgroundPollIntervalMs ?? 2000
		const pollMaxMinutes = this.getBackgroundPollMaxMinutes(model)
		const deadline = Date.now() + pollMaxMinutes * 60_000
		const pollMaxIntervalMs = Math.max(pollIntervalMs, 30_000)
		const pollBackoffFactor = 1.5

		let lastEmittedStatus = options?.lastEmittedStatus
		let pollCount = 0
		let lastPolledStatus: number | undefined
		let nextPollDelayMs = pollIntervalMs

		while (Date.now() <= deadline) {
			if (this.abortController?.signal.aborted) {
				yield { type: "status", mode: "background", status: "canceled", responseId }
				return
			}

			try {
				pollCount += 1
				const pollUrl = this.buildResponsesUrl(`responses/${responseId}`)
				const pollRes = await fetch(pollUrl, {
					method: "GET",
					headers: {
						Authorization: `Bearer ${apiKey}`,
					},
					signal: this.abortController?.signal,
				})

				if (!pollRes.ok) {
					let errorText: string | undefined
					if (pollRes.status === 429) {
						try {
							errorText = await pollRes.text()
						} catch {
							errorText = undefined
						}
					}
					const retryAfterMs = getRetryAfterMs(pollRes.headers)
					const headerKeys = listHeaderKeys(pollRes.headers)
					this.logProvider(
						"error",
						"Polling response not ok",
						{
							method: "GET",
							url: pollUrl,
							status: pollRes.status,
							statusText: pollRes.statusText,
							pollCount,
							responseId,
							headers: pickRateLimitHeaders(pollRes.headers),
							allHeaders: headersToObject(pollRes.headers),
							headerKeys,
							...(typeof errorText === "string"
								? {
										bodyTextLength: errorText.length,
										bodyTextChunks: chunkStringForProviderLog(errorText),
									}
								: {}),
							...(typeof retryAfterMs === "number" ? { retryAfterMs } : {}),
						},
						model.id,
					)
					const statusCode = pollRes.status
					if (statusCode === 401 || statusCode === 403 || statusCode === 404) {
						yield { type: "status", mode: "background", status: "failed", responseId }
						const terminalErr = createTerminalBackgroundError(`Polling failed with status ${statusCode}`)
						;(terminalErr as any).status = statusCode
						throw terminalErr
					}

					// transient; back off and retry
					nextPollDelayMs = Math.min(pollMaxIntervalMs, Math.ceil(nextPollDelayMs * pollBackoffFactor))
					if (typeof retryAfterMs === "number") {
						nextPollDelayMs = Math.min(pollMaxIntervalMs, Math.max(nextPollDelayMs, retryAfterMs))
					}
					await new Promise((r) => setTimeout(r, nextPollDelayMs))
					continue
				}

				let raw: any
				try {
					raw = await pollRes.json()
				} catch {
					nextPollDelayMs = Math.min(pollMaxIntervalMs, Math.ceil(nextPollDelayMs * pollBackoffFactor))
					await new Promise((r) => setTimeout(r, nextPollDelayMs))
					continue
				}

				const resp = raw?.response ?? raw
				const status: string | undefined = resp?.status
				const respId: string | undefined = resp?.id ?? responseId
				const shouldLogStatus = status !== lastEmittedStatus || pollRes.status !== lastPolledStatus
				const pollHeaders = pickRateLimitHeaders(pollRes.headers)
				const pollHeaderKeys = listHeaderKeys(pollRes.headers)
				const pollRetryAfterMs = getRetryAfterMs(pollRes.headers)

				// Capture resolved service tier if present
				if (resp?.service_tier) {
					this.lastServiceTier = resp.service_tier as ServiceTier
				}
				// Capture complete output array (includes reasoning items with encrypted_content)
				if (resp?.output && Array.isArray(resp.output)) {
					this.lastResponseOutput = resp.output
				}
				// Capture top-level response id
				if (typeof respId === "string") {
					this.lastResponseId = respId
				}

				// Emit status transitions
				if (
					status &&
					(status === "queued" ||
						status === "in_progress" ||
						status === "completed" ||
						status === "failed" ||
						status === "canceled")
				) {
					if (status !== lastEmittedStatus) {
						// Reset polling cadence when the response status changes.
						nextPollDelayMs = pollIntervalMs
						if (shouldLogStatus) {
							this.logProvider(
								"status",
								"Polling status update",
								{
									status,
									responseId: respId,
									pollCount,
									httpStatus: pollRes.status,
									headers: pollHeaders,
									headerKeys: pollHeaderKeys,
									...(typeof pollRetryAfterMs === "number" ? { retryAfterMs: pollRetryAfterMs } : {}),
								},
								model.id,
							)
						}
						yield {
							type: "status",
							mode: "background",
							status: status as any,
							...(respId ? { responseId: respId } : {}),
						}
						lastEmittedStatus = status as any
						lastPolledStatus = pollRes.status
					} else if (status === "queued" || status === "in_progress") {
						// Status unchanged: progressively reduce request volume.
						nextPollDelayMs = Math.min(pollMaxIntervalMs, Math.ceil(nextPollDelayMs * pollBackoffFactor))
					}
				}

				if (status === "completed") {
					try {
						const output = resp?.output ?? raw?.output
						const outputText = resp?.output_text ?? raw?.output_text
						const outputItemTypes = Array.isArray(output)
							? Array.from(
									new Set(
										output
											.map((o: any) => (o && typeof o === "object" ? (o as any).type : undefined))
											.filter((t: any) => typeof t === "string"),
									),
								)
							: undefined
						this.logProvider(
							"response",
							"Polling completed response payload summary",
							{
								responseId: respId ?? responseId,
								pollCount,
								wrapped: !!raw?.response,
								rawKeys: raw && typeof raw === "object" ? Object.keys(raw) : undefined,
								respKeys: resp && typeof resp === "object" ? Object.keys(resp) : undefined,
								outputArrayLength: Array.isArray(output) ? output.length : undefined,
								outputItemTypes,
								outputTextLength: typeof outputText === "string" ? outputText.length : undefined,
								usageKeys:
									resp?.usage && typeof resp.usage === "object" ? Object.keys(resp.usage) : undefined,
								headers: pollHeaders,
								headerKeys: pollHeaderKeys,
								...(typeof pollRetryAfterMs === "number" ? { retryAfterMs: pollRetryAfterMs } : {}),
							},
							model.id,
						)
					} catch {
						// Never break polling due to logging failures.
					}

					yield* this.emitFinalResponseOutput(resp, raw, model)
					return
				}

				if (status === "failed" || status === "canceled") {
					const errorObj = resp?.error ?? raw?.error
					const detail: string | undefined = errorObj?.message ?? resp?.error?.message ?? raw?.error?.message
					const httpStatus = pollRes.status
					const httpSuffix = typeof httpStatus === "number" ? ` (HTTP ${httpStatus})` : ""
					const msg = detail
						? `Response ${status}${httpSuffix}: ${detail}`
						: `Response ${status}${httpSuffix}: ${respId || responseId}`

					try {
						this.logProvider(
							"error",
							"Background response returned terminal status",
							{
								status,
								responseId: respId ?? responseId,
								pollCount,
								error: errorObj,
								incompleteDetails: resp?.incomplete_details ?? raw?.incomplete_details,
								httpStatus: pollRes.status,
								headers: pollHeaders,
								headerKeys: pollHeaderKeys,
								...(typeof pollRetryAfterMs === "number" ? { retryAfterMs: pollRetryAfterMs } : {}),
							},
							model.id,
						)
					} catch {
						// Never break polling due to logging failures.
					}

					const terminalErr = createTerminalBackgroundError(msg)
					const lowerDetail = typeof detail === "string" ? detail.toLowerCase() : ""
					const lowerType = typeof errorObj?.type === "string" ? errorObj.type.toLowerCase() : ""
					const lowerCode = typeof errorObj?.code === "string" ? errorObj.code.toLowerCase() : ""
					const isRateLimitLike =
						lowerDetail.includes("too many requests") ||
						lowerDetail.includes("rate limit") ||
						lowerType.includes("rate_limit") ||
						lowerCode.includes("rate_limit")
					;(terminalErr as any).httpStatus = pollRes.status
					;(terminalErr as any).apiErrorCode = typeof errorObj?.code === "string" ? errorObj.code : undefined
					;(terminalErr as any).apiErrorType = typeof errorObj?.type === "string" ? errorObj.type : undefined
					;(terminalErr as any).isRateLimit = isRateLimitLike

					// Some gateways (notably Azure) can return HTTP 200 but a terminal error indicating throttling.
					// When we see a throttling-like terminal status, log the full raw poll response (headers + body) for debugging.
					if (isRateLimitLike) {
						try {
							let rawBodyText = ""
							try {
								rawBodyText = JSON.stringify(raw)
							} catch {
								rawBodyText = String(raw)
							}

							this.logProvider(
								"error",
								"Polling terminal response indicates rate limit (raw)",
								{
									method: "GET",
									url: pollUrl,
									httpStatus: pollRes.status,
									statusText: pollRes.statusText,
									headers: pollHeaders,
									allHeaders: headersToObject(pollRes.headers),
									headerKeys: pollHeaderKeys,
									bodyTextLength: rawBodyText.length,
									bodyTextChunks: chunkStringForProviderLog(rawBodyText),
									responseId: respId ?? responseId,
									pollCount,
								},
								model.id,
							)
						} catch {
							// Never break polling due to logging failures.
						}
					}

					throw terminalErr
				}
			} catch (err: any) {
				this.logProvider("error", "Polling error", { error: err, pollCount, responseId }, model.id)
				// If we've already emitted a terminal status, propagate to consumer to stop polling.
				if (lastEmittedStatus === "failed" || lastEmittedStatus === "canceled") {
					throw err
				}
				if (this.abortController?.signal.aborted) {
					yield { type: "status", mode: "background", status: "canceled", responseId }
					return
				}

				// Classify polling errors and log appropriately
				const statusCode = err?.status ?? err?.response?.status
				const msg = err instanceof Error ? err.message : String(err)

				// Permanent errors: stop polling
				if (statusCode === 401 || statusCode === 403 || statusCode === 404) {
					console.error(`[OpenAiNative][poll] permanent error (status ${statusCode}); stopping: ${msg}`)
					throw createTerminalBackgroundError(`Polling failed with status ${statusCode}: ${msg}`)
				}

				// Rate limit: transient, will retry
				if (statusCode === 429) {
					console.warn(`[OpenAiNative][poll] rate limited; will retry: ${msg}`)
				} else {
					// Other transient/network errors
					console.warn(
						`[OpenAiNative][poll] transient error; will retry${statusCode ? ` (status ${statusCode})` : ""}: ${msg}`,
					)
				}

				// Back off on transient polling errors (including rate limits) to avoid making throttling worse.
				nextPollDelayMs = Math.min(pollMaxIntervalMs, Math.ceil(nextPollDelayMs * pollBackoffFactor))
			}

			// Stop polling immediately on terminal background statuses
			if (lastEmittedStatus === "failed" || lastEmittedStatus === "canceled") {
				throw new Error(`Background polling terminated with status=${lastEmittedStatus} for ${responseId}`)
			}

			await new Promise((r) => setTimeout(r, nextPollDelayMs))
		}

		this.logProvider("error", "Background response polling timed out", { responseId, pollCount }, model.id)
		throw new Error(`Background response polling timed out for ${responseId}`)
	}

	/**
	 * Handles the streaming response from the Responses API.
	 *
	 * This function iterates through the Server-Sent Events (SSE) stream, parses each event,
	 * and yields structured data chunks (`ApiStream`). It handles a wide variety of event types,
	 * including text deltas, reasoning, usage data, and various status/tool events.
	 */
	private async *handleStreamResponse(body: ReadableStream<Uint8Array>, model: OpenAiNativeModel): ApiStream {
		const reader = body.getReader()
		const decoder = new TextDecoder()
		let buffer = ""
		let hasContent = false
		let totalInputTokens = 0
		let totalOutputTokens = 0

		try {
			while (true) {
				// Check if request was aborted
				if (this.abortController?.signal.aborted) {
					break
				}

				const { done, value } = await reader.read()
				if (done) break

				buffer += decoder.decode(value, { stream: true })
				const lines = buffer.split("\n")
				buffer = lines.pop() || ""

				for (const line of lines) {
					if (line.startsWith("data: ")) {
						const data = line.slice(6).trim()
						if (data === "[DONE]") {
							continue
						}

						try {
							const parsed = JSON.parse(data)

							// Skip stale events when resuming a dropped background stream
							if (
								typeof parsed?.sequence_number === "number" &&
								this.resumeCutoffSequence !== undefined &&
								parsed.sequence_number <= this.resumeCutoffSequence
							) {
								continue
							}

							// Record sequence number for cursor tracking
							if (typeof parsed?.sequence_number === "number") {
								this.lastSequenceNumber = parsed.sequence_number
								// Also track for per-request resume capability
								this.currentRequestSequenceNumber = parsed.sequence_number
							}

							// Capture resolved service tier if present
							if (parsed.response?.service_tier) {
								this.lastServiceTier = parsed.response.service_tier as ServiceTier
							}
							// Capture complete output array (includes reasoning items with encrypted_content)
							if (parsed.response?.output && Array.isArray(parsed.response.output)) {
								this.lastResponseOutput = parsed.response.output
							}
							// Capture top-level response id
							if (parsed.response?.id) {
								this.lastResponseId = parsed.response.id as string
								// Also track for per-request resume capability
								this.currentRequestResponseId = parsed.response.id as string
							}

							// Delegate standard event types to the shared processor to avoid duplication
							if (parsed?.type && this.coreHandledEventTypes.has(parsed.type)) {
								for await (const outChunk of this.processEvent(parsed, model)) {
									// Track whether we've emitted any content so fallback handling can decide appropriately
									if (outChunk.type === "text" || outChunk.type === "reasoning") {
										hasContent = true
									}
									yield outChunk
								}
								continue
							}

							// Check if this is a complete response (non-streaming format)
							if (parsed.response && parsed.response.output && Array.isArray(parsed.response.output)) {
								// Handle complete response in the initial event
								for (const outputItem of parsed.response.output) {
									if (outputItem.type === "text" && outputItem.content) {
										for (const content of outputItem.content) {
											if (content.type === "text" && content.text) {
												hasContent = true
												yield {
													type: "text",
													text: content.text,
												}
											}
										}
									}
									// Additionally handle reasoning summaries if present (non-streaming summary output)
									if (outputItem.type === "reasoning" && Array.isArray(outputItem.summary)) {
										for (const summary of outputItem.summary) {
											if (summary?.type === "summary_text" && typeof summary.text === "string") {
												hasContent = true
												yield {
													type: "reasoning",
													text: summary.text,
												}
											}
										}
									}
								}
								// Check for usage in the complete response
								if (parsed.response.usage) {
									const usageData = this.normalizeUsage(parsed.response.usage, model)
									if (usageData) {
										yield usageData
									}
								}
							}
							// Handle streaming delta events for text content
							else if (
								parsed.type === "response.text.delta" ||
								parsed.type === "response.output_text.delta"
							) {
								// Primary streaming event for text deltas
								if (parsed.delta) {
									hasContent = true
									yield {
										type: "text",
										text: parsed.delta,
									}
								}
							} else if (
								parsed.type === "response.text.done" ||
								parsed.type === "response.output_text.done"
							) {
								// Text streaming completed - final text already streamed via deltas
							}
							// Handle reasoning delta events
							else if (
								parsed.type === "response.reasoning.delta" ||
								parsed.type === "response.reasoning_text.delta"
							) {
								// Streaming reasoning content
								if (parsed.delta) {
									hasContent = true
									yield {
										type: "reasoning",
										text: parsed.delta,
									}
								}
							} else if (
								parsed.type === "response.reasoning.done" ||
								parsed.type === "response.reasoning_text.done"
							) {
								// Reasoning streaming completed
							}
							// Handle reasoning summary events
							else if (
								parsed.type === "response.reasoning_summary.delta" ||
								parsed.type === "response.reasoning_summary_text.delta"
							) {
								// Streaming reasoning summary
								if (parsed.delta) {
									hasContent = true
									yield {
										type: "reasoning",
										text: parsed.delta,
									}
								}
							} else if (
								parsed.type === "response.reasoning_summary.done" ||
								parsed.type === "response.reasoning_summary_text.done"
							) {
								// Reasoning summary completed
							}
							// Handle refusal delta events
							else if (parsed.type === "response.refusal.delta") {
								// Model is refusing to answer
								if (parsed.delta) {
									hasContent = true
									yield {
										type: "text",
										text: `[Refusal] ${parsed.delta}`,
									}
								}
							} else if (parsed.type === "response.refusal.done") {
								// Refusal completed
							}
							// Handle audio delta events (for multimodal responses)
							else if (parsed.type === "response.audio.delta") {
								// Audio streaming - we'll skip for now as we focus on text
								// Could be handled in future for voice responses
							} else if (parsed.type === "response.audio.done") {
								// Audio completed
							}
							// Handle audio transcript delta events
							else if (parsed.type === "response.audio_transcript.delta") {
								// Audio transcript streaming
								if (parsed.delta) {
									hasContent = true
									yield {
										type: "text",
										text: parsed.delta,
									}
								}
							} else if (parsed.type === "response.audio_transcript.done") {
								// Audio transcript completed
							}
							// Handle content part events (for structured content)
							else if (parsed.type === "response.content_part.added") {
								// New content part added - could be text, image, etc.
								if (parsed.part?.type === "text" && parsed.part.text) {
									hasContent = true
									yield {
										type: "text",
										text: parsed.part.text,
									}
								}
							} else if (parsed.type === "response.content_part.done") {
								// Content part completed
							}
							// Handle output item events (alternative format)
							else if (parsed.type === "response.output_item.added") {
								// This is where the actual content comes through in some test cases
								if (parsed.item) {
									if (parsed.item.type === "text" && parsed.item.text) {
										hasContent = true
										yield { type: "text", text: parsed.item.text }
									} else if (parsed.item.type === "reasoning" && parsed.item.text) {
										hasContent = true
										yield { type: "reasoning", text: parsed.item.text }
									} else if (parsed.item.type === "message" && parsed.item.content) {
										// Handle message type items
										for (const content of parsed.item.content) {
											if (content.type === "text" && content.text) {
												hasContent = true
												yield { type: "text", text: content.text }
											}
										}
									}
								}
							} else if (parsed.type === "response.output_item.done") {
								// Output item completed
							}
							// Handle function/tool call events
							else if (
								parsed.type === "response.function_call_arguments.delta" ||
								parsed.type === "response.tool_call_arguments.delta" ||
								parsed.type === "response.function_call_arguments.done" ||
								parsed.type === "response.tool_call_arguments.done"
							) {
								// Delegated to processEvent (handles accumulation and completion)
								for await (const outChunk of this.processEvent(parsed, model)) {
									yield outChunk
								}
							}
							// Handle MCP (Model Context Protocol) tool events
							else if (parsed.type === "response.mcp_call_arguments.delta") {
								// MCP tool call arguments streaming
							} else if (parsed.type === "response.mcp_call_arguments.done") {
								// MCP tool call completed
							} else if (parsed.type === "response.mcp_call.in_progress") {
								// MCP tool call in progress
							} else if (
								parsed.type === "response.mcp_call.completed" ||
								parsed.type === "response.mcp_call.failed"
							) {
								// MCP tool call status events
							} else if (parsed.type === "response.mcp_list_tools.in_progress") {
								// MCP list tools in progress
							} else if (
								parsed.type === "response.mcp_list_tools.completed" ||
								parsed.type === "response.mcp_list_tools.failed"
							) {
								// MCP list tools status events
							}
							// Handle web search events
							else if (parsed.type === "response.web_search_call.searching") {
								// Web search in progress
							} else if (parsed.type === "response.web_search_call.in_progress") {
								// Processing web search results
							} else if (parsed.type === "response.web_search_call.completed") {
								// Web search completed
							}
							// Handle code interpreter events
							else if (parsed.type === "response.code_interpreter_call_code.delta") {
								// Code interpreter code streaming
								if (parsed.delta) {
									// Could yield as a special code type if needed
								}
							} else if (parsed.type === "response.code_interpreter_call_code.done") {
								// Code interpreter code completed
							} else if (parsed.type === "response.code_interpreter_call.interpreting") {
								// Code interpreter running
							} else if (parsed.type === "response.code_interpreter_call.in_progress") {
								// Code execution in progress
							} else if (parsed.type === "response.code_interpreter_call.completed") {
								// Code interpreter completed
							}
							// Handle file search events
							else if (parsed.type === "response.file_search_call.searching") {
								// File search in progress
							} else if (parsed.type === "response.file_search_call.in_progress") {
								// Processing file search results
							} else if (parsed.type === "response.file_search_call.completed") {
								// File search completed
							}
							// Handle image generation events
							else if (parsed.type === "response.image_gen_call.generating") {
								// Image generation in progress
							} else if (parsed.type === "response.image_gen_call.in_progress") {
								// Processing image generation
							} else if (parsed.type === "response.image_gen_call.partial_image") {
								// Image partially generated
							} else if (parsed.type === "response.image_gen_call.completed") {
								// Image generation completed
							}
							// Handle computer use events
							else if (
								parsed.type === "response.computer_tool_call.output_item" ||
								parsed.type === "response.computer_tool_call.output_screenshot"
							) {
								// Computer use tool events
							}
							// Handle annotation events
							else if (
								parsed.type === "response.output_text_annotation.added" ||
								parsed.type === "response.text_annotation.added"
							) {
								// Text annotation events - could be citations, references, etc.
							}
							// Handle error events
							else if (parsed.type === "response.error" || parsed.type === "error") {
								// Error event from the API
								if (parsed.error || parsed.message) {
									const errMsg = `Responses API stream error (raw): ${data}`
									this.logProvider(
										"error",
										"Responses API stream error event",
										{
											eventType: parsed.type,
											responseId: parsed.response?.id ?? this.currentRequestResponseId,
											sequence_number: parsed.sequence_number,
											error: parsed.error,
											message: parsed.message,
											errorCode: parsed.error?.code,
											errorType: parsed.error?.type,
											errorParam: parsed.error?.param,
											errorStatus:
												parsed.error?.status ??
												parsed.error?.status_code ??
												parsed.status ??
												parsed.status_code,
										},
										model.id,
									)
									// For background mode, treat as terminal to avoid futile resume attempts
									if (this.currentRequestIsBackground) {
										// Surface a failed status for UI lifecycle before terminating
										yield {
											type: "status",
											mode: "background",
											status: "failed",
											...(parsed.response?.id ? { responseId: parsed.response.id } : {}),
										}
										throw createTerminalBackgroundError(errMsg)
									}
									// Non-background: propagate as a standard error
									throw new Error(errMsg)
								}
							}
							// Handle incomplete event
							else if (parsed.type === "response.incomplete") {
								// Response was incomplete - might need to handle specially
							}
							// Handle queued event
							else if (parsed.type === "response.queued") {
								yield {
									type: "status",
									mode: this.currentRequestIsBackground ? "background" : undefined,
									status: "queued",
									...(parsed.response?.id ? { responseId: parsed.response.id } : {}),
								}
							}
							// Handle in_progress event
							else if (parsed.type === "response.in_progress") {
								yield {
									type: "status",
									mode: this.currentRequestIsBackground ? "background" : undefined,
									status: "in_progress",
									...(parsed.response?.id ? { responseId: parsed.response.id } : {}),
								}
							}
							// Handle failed event
							else if (parsed.type === "response.failed") {
								// Emit failed status for UI lifecycle
								yield {
									type: "status",
									mode: this.currentRequestIsBackground ? "background" : undefined,
									status: "failed",
									...(parsed.response?.id ? { responseId: parsed.response.id } : {}),
								}
								// Response failed
								if (parsed.error || parsed.message) {
									throw createTerminalBackgroundError(
										`Response failed: ${parsed.error?.message || parsed.message || "Unknown failure"}`,
									)
								}
							} else if (parsed.type === "response.completed" || parsed.type === "response.done") {
								// Capture resolved service tier if present
								if (parsed.response?.service_tier) {
									this.lastServiceTier = parsed.response.service_tier as ServiceTier
								}
								// Capture top-level response id
								if (parsed.response?.id) {
									this.lastResponseId = parsed.response.id as string
								}
								// Capture complete output array (includes reasoning items with encrypted_content)
								if (parsed.response?.output && Array.isArray(parsed.response.output)) {
									this.lastResponseOutput = parsed.response.output
								}

								// Emit completed status for UI lifecycle
								yield {
									type: "status",
									mode: this.currentRequestIsBackground ? "background" : undefined,
									status: "completed",
									...(parsed.response?.id ? { responseId: parsed.response.id } : {}),
								}
								// Clear background marker on completion
								this.currentRequestIsBackground = undefined

								// Check if the done event contains the complete output (as a fallback)
								if (
									!hasContent &&
									parsed.response &&
									parsed.response.output &&
									Array.isArray(parsed.response.output)
								) {
									for (const outputItem of parsed.response.output) {
										if (outputItem.type === "message" && outputItem.content) {
											for (const content of outputItem.content) {
												if (content.type === "output_text" && content.text) {
													hasContent = true
													yield {
														type: "text",
														text: content.text,
													}
												}
											}
										}
										// Also surface reasoning summaries if present in the final output
										if (outputItem.type === "reasoning" && Array.isArray(outputItem.summary)) {
											for (const summary of outputItem.summary) {
												if (
													summary?.type === "summary_text" &&
													typeof summary.text === "string"
												) {
													hasContent = true
													yield {
														type: "reasoning",
														text: summary.text,
													}
												}
											}
										}
									}
								}

								// Usage for done/completed is already handled by processEvent in the SDK path.
								// For SSE path, usage often arrives separately; avoid double-emitting here.
							}
							// These are structural or status events, we can just log them at a lower level or ignore.
							else if (
								parsed.type === "response.created" ||
								parsed.type === "response.in_progress" ||
								parsed.type === "response.output_item.done" ||
								parsed.type === "response.content_part.added" ||
								parsed.type === "response.content_part.done"
							) {
								// Status events - no action needed
							}
							// Fallback for older formats or unexpected responses
							else if (parsed.choices?.[0]?.delta?.content) {
								hasContent = true
								yield {
									type: "text",
									text: parsed.choices[0].delta.content,
								}
							}
							// Additional fallback: some events place text under 'item.text' even if type isn't matched above
							else if (
								parsed.item &&
								typeof parsed.item.text === "string" &&
								parsed.item.text.length > 0
							) {
								hasContent = true
								yield {
									type: "text",
									text: parsed.item.text,
								}
							} else if (parsed.usage) {
								// Handle usage if it arrives in a separate, non-completed event
								const usageData = this.normalizeUsage(parsed.usage, model)
								if (usageData) {
									yield usageData
								}
							}
						} catch (e) {
							// Only ignore JSON parsing errors, re-throw actual API errors
							if (!(e instanceof SyntaxError)) {
								throw e
							}
						}
					}
					// Also try to parse non-SSE formatted lines
					else if (line.trim() && !line.startsWith(":")) {
						try {
							const parsed = JSON.parse(line)

							// Try to extract content from various possible locations
							if (parsed.content || parsed.text || parsed.message) {
								hasContent = true
								yield {
									type: "text",
									text: parsed.content || parsed.text || parsed.message,
								}
							}
						} catch {
							// Not JSON, might be plain text - ignore
						}
					}
				}
			}

			// If we didn't get any content, don't throw - the API might have returned an empty response
			// This can happen in certain edge cases and shouldn't break the flow
		} catch (error) {
			if (error instanceof Error) {
				// Preserve terminal background errors so callers can avoid resume attempts
				if ((error as any).isTerminalBackgroundError) {
					throw error
				}
				throw new Error(`Error processing response stream: ${error.message}`)
			}
			throw new Error("Unexpected error processing response stream")
		} finally {
			reader.releaseLock()
		}
	}

	/**
	 * Attempt to resume a dropped background stream; if resume fails, fall back to polling.
	 */
	private async *attemptResumeOrPoll(responseId: string, lastSeq: number, model: OpenAiNativeModel): ApiStream {
		this.logProvider(
			"retry",
			"Attempting resume for background response",
			{ responseId, lastSequenceNumber: lastSeq },
			model.id,
		)

		// Emit reconnecting status
		yield {
			type: "status",
			mode: "background",
			status: "reconnecting",
			responseId,
		}

		const apiKey = this.options.openAiNativeApiKey ?? "not-provided"
		const configuredResumeMaxRetries = this.options.openAiNativeBackgroundResumeMaxRetries
		const resumeMaxRetries =
			isAzureOpenAiBaseUrl(this.options.openAiNativeBaseUrl) && configuredResumeMaxRetries === undefined
				? 0
				: (configuredResumeMaxRetries ?? 3)
		const resumeBaseDelayMs = this.options.openAiNativeBackgroundResumeBaseDelayMs ?? 1000

		// Try streaming resume with exponential backoff
		for (let attempt = 0; attempt < resumeMaxRetries; attempt++) {
			try {
				const resumeUrl = this.buildResponsesUrl(
					`responses/${responseId}?stream=true&starting_after=${lastSeq}`,
				)
				this.logProvider("request", "Resume request", { attempt: attempt + 1, resumeUrl }, model.id)
				const res = await fetch(resumeUrl, {
					method: "GET",
					headers: {
						Authorization: `Bearer ${apiKey}`,
						Accept: "text/event-stream",
					},
					signal: this.abortController?.signal,
				})
				const retryAfterMs = getRetryAfterMs(res.headers)

				this.logProvider(
					"response",
					"Resume response received",
					{
						attempt: attempt + 1,
						status: res.status,
						ok: res.ok,
						headers: pickRateLimitHeaders(res.headers),
						...(typeof retryAfterMs === "number" ? { retryAfterMs } : {}),
					},
					model.id,
				)

				if (!res.ok) {
					const status = res.status
					if (status === 429) {
						let bodyText: string | undefined
						try {
							bodyText = await res.text()
						} catch {
							bodyText = undefined
						}
						this.logProvider(
							"error",
							"Resume HTTP 429 response (raw)",
							{
								attempt: attempt + 1,
								method: "GET",
								url: resumeUrl,
								status,
								statusText: res.statusText,
								headers: pickRateLimitHeaders(res.headers),
								allHeaders: headersToObject(res.headers),
								headerKeys: listHeaderKeys(res.headers),
								...(typeof bodyText === "string"
									? {
											bodyTextLength: bodyText.length,
											bodyTextChunks: chunkStringForProviderLog(bodyText),
										}
									: {}),
							},
							model.id,
						)
					}
					if (status === 401 || status === 403) {
						yield {
							type: "status",
							mode: "background",
							status: "failed",
							responseId,
						}

						const terminalErr = createTerminalBackgroundError(`Resume request failed (${status})`)
						;(terminalErr as any).status = status
						throw terminalErr
					}
					if (status === 404 || status === 408 || status === 504) {
						// Some gateways may not support the resume streaming endpoint (404) or may time out (408/504)
						// even when the stored response exists; fall back to polling instead of failing the request.
						break
					}

					throw new Error(`Resume request failed (${status})`)
				}
				if (!res.body) {
					throw new Error("Resume request failed (no body)")
				}

				this.resumeCutoffSequence = lastSeq

				// Handshake accepted: immediately switch UI from reconnecting -> in_progress
				yield {
					type: "status",
					mode: "background",
					status: "in_progress",
					responseId,
				}

				try {
					for await (const chunk of this.handleStreamResponse(res.body, model)) {
						// Avoid double-emitting in_progress if the inner handler surfaces it
						if (chunk.type === "status" && (chunk as any).status === "in_progress") {
							continue
						}
						yield chunk
					}
					// Successful resume
					this.resumeCutoffSequence = undefined
					return
				} catch (e) {
					// Resume stream failed mid-flight; reset and throw to retry
					this.resumeCutoffSequence = undefined
					throw e
				}
			} catch (err: any) {
				this.logProvider("error", "Resume attempt failed", { attempt: attempt + 1, error: err }, model.id)
				const delay = resumeBaseDelayMs * Math.pow(2, attempt)
				const msg = err instanceof Error ? err.message : String(err)

				if (isTerminalBackgroundError(err)) {
					const statusCode = (err as any).status
					console.error(
						`[OpenAiNative][resume] terminal background error on attempt ${attempt + 1}${
							statusCode ? ` (status ${statusCode})` : ""
						}: ${msg}`,
					)
					// Only fall back to polling when resume is unsupported (404). Otherwise, surface the terminal error
					// so we don't add additional requests (polling) when the response is already failed.
					if (statusCode === 404) {
						break
					}
					throw err
				}

				// Otherwise retry with backoff (transient failure)
				console.warn(`[OpenAiNative][resume] attempt ${attempt + 1} failed; retrying in ${delay}ms: ${msg}`)
				if (delay > 0) {
					await new Promise((r) => setTimeout(r, delay))
				}
			}
		}

		// Resume failed - begin polling fallback
		this.logProvider("status", "Switching to polling fallback", { responseId }, model.id)
		yield {
			type: "status",
			mode: "background",
			status: "polling",
			responseId,
		}

		yield* this.pollBackgroundResponse(responseId, model)
		return
	}

	/**
	 * Shared processor for Responses API events.
	 */
	private async *processEvent(event: any, model: OpenAiNativeModel): ApiStream {
		// Capture resolved service tier when available
		if (event?.response?.service_tier) {
			this.lastServiceTier = event.response.service_tier as ServiceTier
		}
		// Capture complete output array (includes reasoning items with encrypted_content)
		if (event?.response?.output && Array.isArray(event.response.output)) {
			this.lastResponseOutput = event.response.output
		}
		// Capture top-level response id
		if (event?.response?.id) {
			this.lastResponseId = event.response.id as string
			// Also track for per-request resume capability
			this.currentRequestResponseId = event.response.id as string
		}
		// Record sequence number for cursor tracking
		if (typeof event?.sequence_number === "number") {
			this.lastSequenceNumber = event.sequence_number
			// Also track for per-request resume capability
			this.currentRequestSequenceNumber = event.sequence_number
		}

		// Handle explicit error events emitted in-stream by the API/SDK.
		// Treat background-mode errors as terminal to avoid futile resume/poll fallbacks.
		if (event?.type === "response.error" || event?.type === "error") {
			let raw = ""
			try {
				raw = JSON.stringify(event)
			} catch {
				raw = String(event)
			}
			const errMsg = `Responses API stream error (raw): ${raw}`
			this.logProvider(
				"error",
				"Streaming error event",
				{
					eventType: event?.type,
					responseId: event?.response?.id ?? this.currentRequestResponseId,
					sequence: event?.sequence_number,
					error: event?.error,
					message: event?.message,
					errorCode: event?.error?.code,
					errorType: event?.error?.type,
					errorParam: event?.error?.param,
					errorStatus:
						event?.error?.status ?? event?.error?.status_code ?? event?.status ?? event?.status_code,
				},
				model.id,
			)

			if (this.currentRequestIsBackground) {
				yield {
					type: "status",
					mode: "background",
					status: "failed",
					...(event?.response?.id ? { responseId: event.response.id } : {}),
				}
				throw createTerminalBackgroundError(errMsg)
			}

			throw new Error(errMsg)
		}

		// Map lifecycle events to status chunks
		const statusMap: Record<string, "queued" | "in_progress" | "completed" | "failed" | "canceled"> = {
			"response.queued": "queued",
			"response.in_progress": "in_progress",
			"response.completed": "completed",
			"response.done": "completed",
			"response.failed": "failed",
			"response.canceled": "canceled",
		}
		const mappedStatus = statusMap[event?.type as string]
		if (mappedStatus) {
			this.logProvider(
				"status",
				"Streaming status event",
				{
					status: mappedStatus,
					eventType: event?.type,
					responseId: event?.response?.id,
					sequence: event?.sequence_number,
				},
				model.id,
			)
			yield {
				type: "status",
				mode: this.currentRequestIsBackground ? "background" : undefined,
				status: mappedStatus,
				...(event?.response?.id ? { responseId: event.response.id } : {}),
			}
			// Clear background flag for terminal statuses
			if (mappedStatus === "completed" || mappedStatus === "failed" || mappedStatus === "canceled") {
				this.currentRequestIsBackground = undefined
			}
			// Throw terminal error to integrate with standard failure path (surfaced in UI)
			if (mappedStatus === "failed" || mappedStatus === "canceled") {
				const msg = (event as any)?.error?.message || (event as any)?.message || `Response ${mappedStatus}`
				throw createTerminalBackgroundError(msg)
			}
			// Do not return; allow further handling (e.g., usage on done/completed)
		}

		// Handle known streaming text deltas
		if (event?.type === "response.text.delta" || event?.type === "response.output_text.delta") {
			if (event?.delta) {
				yield { type: "text", text: event.delta }
			}
			return
		}

		// Handle reasoning deltas (including summary variants)
		if (
			event?.type === "response.reasoning.delta" ||
			event?.type === "response.reasoning_text.delta" ||
			event?.type === "response.reasoning_summary.delta" ||
			event?.type === "response.reasoning_summary_text.delta"
		) {
			if (event?.delta) {
				yield { type: "reasoning", text: event.delta }
			}
			return
		}

		// Handle refusal deltas
		if (event?.type === "response.refusal.delta") {
			if (event?.delta) {
				yield { type: "text", text: `[Refusal] ${event.delta}` }
			}
			return
		}

		// Handle tool/function call deltas - emit as partial chunks
		if (
			event?.type === "response.tool_call_arguments.delta" ||
			event?.type === "response.function_call_arguments.delta"
		) {
			// Emit partial chunks directly - NativeToolCallParser handles state management
			const callId = event.call_id || event.tool_call_id || event.id
			const name = event.name || event.function_name
			const args = event.delta || event.arguments

			yield {
				type: "tool_call_partial",
				index: event.index ?? 0,
				id: callId,
				name,
				arguments: args,
			}
			return
		}

		// Handle tool/function call completion events
		if (
			event?.type === "response.tool_call_arguments.done" ||
			event?.type === "response.function_call_arguments.done"
		) {
			// Tool call complete - no action needed, NativeToolCallParser handles completion
			return
		}

		// Handle output item additions/completions (SDK or Responses API alternative format)
		if (event?.type === "response.output_item.added" || event?.type === "response.output_item.done") {
			const item = event?.item
			if (item) {
				if (item.type === "text" && item.text) {
					yield { type: "text", text: item.text }
				} else if (item.type === "reasoning" && item.text) {
					yield { type: "reasoning", text: item.text }
				} else if (item.type === "message" && Array.isArray(item.content)) {
					for (const content of item.content) {
						// Some implementations send 'text'; others send 'output_text'
						if ((content?.type === "text" || content?.type === "output_text") && content?.text) {
							yield { type: "text", text: content.text }
						}
					}
				} else if (
					(item.type === "function_call" || item.type === "tool_call") &&
					event.type === "response.output_item.done" // Only handle done events for tool calls to ensure arguments are complete
				) {
					// Handle complete tool/function call item
					// Emit as tool_call for backward compatibility with non-streaming tool handling
					const callId = item.call_id || item.tool_call_id || item.id
					if (callId) {
						const args = item.arguments || item.function?.arguments || item.function_arguments
						yield {
							type: "tool_call",
							id: callId,
							name: item.name || item.function?.name || item.function_name || "",
							arguments: typeof args === "string" ? args : "{}",
						}
					}
				}
			}
			return
		}

		// Completion events that may carry usage
		if (event?.type === "response.done" || event?.type === "response.completed") {
			const usage = event?.response?.usage || event?.usage || undefined
			const usageData = this.normalizeUsage(usage, model)
			if (usageData) {
				yield usageData
			}
			return
		}

		// Fallbacks for older formats or unexpected objects
		if (event?.choices?.[0]?.delta?.content) {
			yield { type: "text", text: event.choices[0].delta.content }
			return
		}

		if (event?.usage) {
			const usageData = this.normalizeUsage(event.usage, model)
			if (usageData) {
				yield usageData
			}
		}
	}

	private getReasoningEffort(model: OpenAiNativeModel): ReasoningEffortExtended | undefined {
		// Single source of truth: user setting overrides, else model default (from types).
		const selected = (this.options.reasoningEffort as any) ?? (model.info.reasoningEffort as any)
		return selected && selected !== "disable" ? (selected as any) : undefined
	}

	/**
	 * Returns the appropriate prompt cache retention policy for the given model, if any.
	 *
	 * The policy is driven by ModelInfo.promptCacheRetention so that model-specific details
	 * live in the shared types layer rather than this provider. When set to "24h" and the
	 * model supports prompt caching, extended prompt cache retention is requested.
	 */
	private getPromptCacheRetention(model: OpenAiNativeModel): "24h" | undefined {
		if (!model.info.supportsPromptCache) return undefined

		if (model.info.promptCacheRetention === "24h") {
			return "24h"
		}

		return undefined
	}

	/**
	 * Returns a shallow-cloned ModelInfo with pricing overridden for the given tier, if available.
	 * If no tier or no overrides exist, the original ModelInfo is returned.
	 */
	private applyServiceTierPricing(info: ModelInfo, tier?: ServiceTier): ModelInfo {
		if (!tier || tier === "default") return info

		// Find the tier with matching name in the tiers array
		const tierInfo = info.tiers?.find((t) => t.name === tier)
		if (!tierInfo) return info

		return {
			...info,
			inputPrice: tierInfo.inputPrice ?? info.inputPrice,
			outputPrice: tierInfo.outputPrice ?? info.outputPrice,
			cacheReadsPrice: tierInfo.cacheReadsPrice ?? info.cacheReadsPrice,
			cacheWritesPrice: tierInfo.cacheWritesPrice ?? info.cacheWritesPrice,
		}
	}

	// Removed isResponsesApiModel method as ALL models now use the Responses API

	override getModel() {
		const modelId = this.options.apiModelId

		let id =
			modelId && modelId in openAiNativeModels ? (modelId as OpenAiNativeModelId) : openAiNativeDefaultModelId

		const info: ModelInfo = openAiNativeModels[id]

		const params = getModelParams({
			format: "openai",
			modelId: id,
			model: info,
			settings: this.options,
			defaultTemperature: OPENAI_NATIVE_DEFAULT_TEMPERATURE,
		})

		// Reasoning effort inclusion is handled by getModelParams/getOpenAiReasoning.
		// Do not re-compute or filter efforts here.

		// The o3 models are named like "o3-mini-[reasoning-effort]", which are
		// not valid model ids, so we need to strip the suffix.
		return { id: id.startsWith("o3-mini") ? "o3-mini" : id, info, ...params, verbosity: params.verbosity }
	}

	/**
	 * Extracts encrypted_content and id from the first reasoning item in the output array.
	 * This is the minimal data needed for stateless API continuity.
	 *
	 * @returns Object with encrypted_content and id, or undefined if not available
	 */
	getEncryptedContent(): { encrypted_content: string; id?: string } | undefined {
		if (!this.lastResponseOutput) return undefined

		// Find the first reasoning item with encrypted_content
		const reasoningItem = this.lastResponseOutput.find(
			(item) => item.type === "reasoning" && item.encrypted_content,
		)

		if (!reasoningItem?.encrypted_content) return undefined

		return {
			encrypted_content: reasoningItem.encrypted_content,
			...(reasoningItem.id ? { id: reasoningItem.id } : {}),
		}
	}

	getResponseId(): string | undefined {
		return this.lastResponseId
	}

	/**
	 * Gets the last sequence number observed from streaming events.
	 * @returns The sequence number, or undefined if not available yet
	 */
	getLastSequenceNumber(): number | undefined {
		return this.lastSequenceNumber
	}

	/**
	 * Seeds `previous_response_id` chaining for stored conversations.
	 * Used after task resume/restart to continue a stored Responses thread without replaying history.
	 * @param responseId The stored response ID to chain from
	 */
	setResponseId(responseId: string): void {
		this.previousResponseId = responseId
	}

	async completePrompt(prompt: string): Promise<string> {
		// Create AbortController for cancellation
		this.abortController = new AbortController()

		try {
			const model = this.getModel()
			const { verbosity, reasoning } = model

			// Resolve reasoning effort for models that support it
			const reasoningEffort = this.getReasoningEffort(model)

			// Build request body for Responses API
			const requestBody: any = {
				model: model.id,
				input: [
					{
						role: "user",
						content: [{ type: "input_text", text: prompt }],
					},
				],
				stream: false, // Non-streaming for completePrompt
				store: false, // Don't store prompt completions
				// Only include encrypted reasoning content when reasoning effort is set
				...(reasoningEffort ? { include: ["reasoning.encrypted_content"] } : {}),
			}

			// Include service tier if selected and supported
			const requestedTier = (this.options.openAiNativeServiceTier as ServiceTier | undefined) || undefined
			const allowedTierNames = new Set(model.info.tiers?.map((t) => t.name).filter(Boolean) || [])
			if (requestedTier && (requestedTier === "default" || allowedTierNames.has(requestedTier))) {
				requestBody.service_tier = requestedTier
			}

			// Add reasoning if supported
			if (reasoningEffort) {
				requestBody.reasoning = {
					effort: reasoningEffort,
					...(this.options.enableResponsesReasoningSummary ? { summary: "auto" as const } : {}),
				}
			}

			// Only include temperature if the model supports it
			if (model.info.supportsTemperature !== false) {
				requestBody.temperature = this.options.modelTemperature ?? OPENAI_NATIVE_DEFAULT_TEMPERATURE
			}

			// Include max_output_tokens if available
			const maxOutputTokens = this.getMaxOutputTokensForRequest(model)
			if (maxOutputTokens) {
				requestBody.max_output_tokens = maxOutputTokens
			}

			// Include text.verbosity only when the model explicitly supports it
			if (model.info.supportsVerbosity === true) {
				requestBody.text = { verbosity: (verbosity || "medium") as VerbosityLevel }
			}

			// Enable extended prompt cache retention for eligible models
			const promptCacheRetention = this.getPromptCacheRetention(model)
			if (promptCacheRetention) {
				requestBody.prompt_cache_retention = promptCacheRetention
			}

			// Make the non-streaming request
			const response = await (this.client as any).responses.create(requestBody, {
				signal: this.abortController.signal,
			})

			// Extract text from the response
			if (response?.output && Array.isArray(response.output)) {
				for (const outputItem of response.output) {
					if (outputItem.type === "message" && outputItem.content) {
						for (const content of outputItem.content) {
							if (content.type === "output_text" && content.text) {
								return content.text
							}
						}
					}
				}
			}

			// Fallback: check for direct text in response
			if (response?.text) {
				return response.text
			}

			return ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`OpenAI Native completion error: ${error.message}`)
			}
			throw error
		} finally {
			this.abortController = undefined
		}
	}
}
