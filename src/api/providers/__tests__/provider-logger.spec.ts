import { describe, it, expect, beforeEach, afterEach } from "vitest"

import { logProviderEvent, setProviderLogger } from "../provider-logger"

describe("provider-logger sanitize()", () => {
	beforeEach(() => {
		setProviderLogger(undefined)
	})

	afterEach(() => {
		setProviderLogger(undefined)
	})

	it("truncates long strings but preserves array/object structure", () => {
		const entries: any[] = []
		setProviderLogger((entry) => entries.push(entry))

		const long = "x".repeat(350)
		const data = {
			input: Array.from({ length: 25 }, (_v, i) => ({ i, text: long })),
			nested: { a: { b: { c: { d: { e: { f: { g: "ok" } } } } } } },
		}

		logProviderEvent({ provider: "test", stage: "request", data })

		expect(entries).toHaveLength(1)
		expect(entries[0].data.input).toHaveLength(25)
		expect(entries[0].data.input[0].text.length).toBeLessThan(long.length)
		expect(entries[0].data.nested.a.b.c.d.e.f.g).toBe("ok")
	})

	it("redacts common secret keys", () => {
		const entries: any[] = []
		setProviderLogger((entry) => entries.push(entry))

		logProviderEvent({
			provider: "test",
			stage: "request",
			data: {
				headers: {
					Authorization: "Bearer secret",
				},
				apiKey: "also-secret",
			},
		})

		expect(entries[0].data.headers.Authorization).toBe("<redacted>")
		expect(entries[0].data.apiKey).toBe("<redacted>")
	})

	it("replaces data-URIs with a length marker", () => {
		const entries: any[] = []
		setProviderLogger((entry) => entries.push(entry))

		const uri = `data:text/plain;base64,${"a".repeat(500)}`
		logProviderEvent({ provider: "test", stage: "request", data: { uri } })

		expect(entries[0].data.uri).toBe(`<data-uri length=${uri.length}>`)
	})

	it("handles circular references safely", () => {
		const entries: any[] = []
		setProviderLogger((entry) => entries.push(entry))

		const obj: any = { ok: true }
		obj.self = obj

		logProviderEvent({ provider: "test", stage: "request", data: obj })

		expect(entries[0].data.ok).toBe(true)
		expect(entries[0].data.self).toBe("[circular]")
	})

	it("includes formatted error details", () => {
		const entries: any[] = []
		setProviderLogger((entry) => entries.push(entry))

		const err = new Error("boom")
		;(err as any).status = 429

		logProviderEvent({ provider: "test", stage: "error", error: err })

		expect(entries[0].error.name).toBe("Error")
		expect(entries[0].error.message).toBe("boom")
		expect(entries[0].error.status).toBe(429)
	})
})
