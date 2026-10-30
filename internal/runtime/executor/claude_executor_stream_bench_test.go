package executor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
)

// buildClaudeAgentHistoryBenchmarkBody builds a native Claude Messages request
// shaped like a long coding-agent turn: a large system prompt, a few dozen
// tools, structured effort in output_config, and a history of thinking +
// tool_use / tool_result rounds. turns controls the history length.
func buildClaudeAgentHistoryBenchmarkBody(tb testing.TB, turns int) []byte {
	tb.Helper()
	filler := func(n int, seed string) string {
		var sb strings.Builder
		for sb.Len() < n {
			sb.WriteString(seed)
			sb.WriteString(" line of tool output, source code or prose; ")
		}
		return sb.String()[:n]
	}
	tools := make([]map[string]any, 0, 30)
	for i := 0; i < 30; i++ {
		tools = append(tools, map[string]any{
			"name":        fmt.Sprintf("agent_tool_%02d", i),
			"description": filler(600, fmt.Sprintf("tool %d", i)),
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string", "description": filler(120, "arg")},
					"workdir": map[string]any{"type": "string"},
				},
				"required": []string{"command"},
			},
		})
	}
	messages := []map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": filler(2000, "task")}}}}
	for i := 0; i < turns; i++ {
		id := fmt.Sprintf("toolu_%06d", i)
		messages = append(messages,
			map[string]any{"role": "assistant", "content": []map[string]any{
				{"type": "thinking", "thinking": filler(300, "think"), "signature": filler(400, "sig")},
				{"type": "tool_use", "id": id, "name": fmt.Sprintf("agent_tool_%02d", i%30), "input": map[string]any{"command": filler(150, "cmd")}},
			}},
			map[string]any{"role": "user", "content": []map[string]any{
				{"type": "tool_result", "tool_use_id": id, "content": []map[string]any{{"type": "text", "text": filler(1700, fmt.Sprintf("out %d", i))}}},
			}},
		)
	}
	last := messages[len(messages)-1]["content"].([]map[string]any)
	last[0]["cache_control"] = map[string]any{"type": "ephemeral"}
	body := map[string]any{
		"model":      "claude-opus-5-5",
		"max_tokens": 32000,
		"stream":     true,
		"system": []map[string]any{
			{"type": "text", "text": filler(30000, "system")},
			{"type": "text", "text": filler(4000, "env"), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"tools":         tools,
		"messages":      messages,
		"thinking":      map[string]any{"type": "adaptive"},
		"output_config": map[string]any{"effort": "high"},
	}
	raw, err := json.Marshal(body)
	if err != nil {
		tb.Fatal(err)
	}
	return raw
}

// BenchmarkClaudeExecutorStreamOAuthAgentHistory measures the CPU the executor
// spends preparing a cloaked OAuth streaming request before it reaches the
// upstream, which a long agent history turns into per-request latency.
func BenchmarkClaudeExecutorStreamOAuthAgentHistory(b *testing.B) {
	gin.SetMode(gin.TestMode)
	stream := strings.Join([]string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
		`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"
	transport := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		_, _ = io.Copy(io.Discard, req.Body)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(stream)),
			Request:    req,
		}, nil
	})
	for _, turns := range []int{50, 300} {
		payload := buildClaudeAgentHistoryBenchmarkBody(b, turns)
		b.Run(fmt.Sprintf("%dturns_%dKiB", turns, len(payload)>>10), func(b *testing.B) {
			ex := NewClaudeExecutor(&config.Config{})
			auth := &cliproxyauth.Auth{ID: "bench-oauth", Attributes: map[string]string{"api_key": "sk-ant-oat-bench"},
				Metadata: map[string]any{"account_uuid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}}
			opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude, ResponseFormat: sdktranslator.FormatClaude,
				Stream: true, OriginalRequest: payload}
			b.ReportAllocs()
			b.SetBytes(int64(len(payload)))
			for b.Loop() {
				ginCtx, _ := gin.CreateTestContext(nil)
				ginCtx.Request, _ = http.NewRequest(http.MethodPost, "http://example.invalid/v1/messages", nil)
				ctx := context.WithValue(context.WithValue(context.Background(), "gin", ginCtx), "cliproxy.roundtripper", http.RoundTripper(transport))
				req := cliproxyexecutor.Request{Model: "claude-opus-5-5", Payload: append([]byte(nil), payload...)}
				result, err := ex.ExecuteStream(ctx, auth, req, opts)
				if err != nil {
					b.Fatal(err)
				}
				for chunk := range result.Chunks {
					if chunk.Err != nil {
						b.Fatal(chunk.Err)
					}
				}
			}
		})
	}
}
