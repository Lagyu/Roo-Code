export function markdownFormattingSection(): string {
	return `====

MARKDOWN RULES

When referencing a specific file path or symbol with a known location, format it as a clickable link like [\`name\`](relative/file/path.ext:line). Only include a line number when you actually know it; do not invent line numbers. If no concrete location is known, use plain text. This applies to all markdown responses and attempt_completion.`
}
