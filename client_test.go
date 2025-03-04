package openaiclient

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

// FakeClient is a simple fake httpClient implementation.
type FakeClient struct {
	DoFunc func(req *http.Request) (*http.Response, error)
}

func (f *FakeClient) Do(req *http.Request) (*http.Response, error) {
	return f.DoFunc(req)
}

// SequentialFakeClient allows simulating a sequence of HTTP responses.
type SequentialFakeClient struct {
	Responses []*http.Response
	CallCount int
}

func (s *SequentialFakeClient) Do(req *http.Request) (*http.Response, error) {
	if s.CallCount >= len(s.Responses) {
		return nil, errors.New("no more responses")
	}
	resp := s.Responses[s.CallCount]
	s.CallCount++
	return resp, nil
}

// fakeResponse is a helper to create an HTTP response with a given body.
func fakeResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// TestNewOpenAIDefaults verifies that NewOpenAI falls back to environment variables.
func TestNewOpenAIDefaults(t *testing.T) {
	// Use t.Setenv (Go 1.17+) to set env variables for the duration of this test.
	t.Setenv("OPENAI_BASE_URL", "http://env-url.com")
	t.Setenv("OPENAI_API_KEY", "env-key")

	client := NewOpenAI("", "")
	if client.baseUrl != "http://env-url.com" {
		t.Errorf("expected baseUrl to be 'http://env-url.com', got '%s'", client.baseUrl)
	}
	if client.key != "env-key" {
		t.Errorf("expected api key to be 'env-key', got '%s'", client.key)
	}
}

// TestGetEmbedding_Success simulates a successful GetEmbedding call.
func TestGetEmbedding_Success(t *testing.T) {
	// Create a fake embedding response.
	embeddingResponse := GetEmbeddingResponse{
		Object: "embedding",
		Data: []EmbeddingObject{
			{
				Object:    "embedding",
				Index:     0,
				Embedding: []float64{0.1, 0.2, 0.3},
				Model:     "test-model",
			},
		},
		Usage: &LLMUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
	respBody, _ := json.Marshal(embeddingResponse)
	fakeClient := &FakeClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return fakeResponse(200, string(respBody)), nil
		},
	}

	client := NewOpenAI("http://example.com", "test-key")
	client.client = fakeClient // override with our fake client

	payload := GetEmbeddingPayload{
		Model: "test-model",
		Input: "Hello",
	}

	embedding, err := client.GetEmbedding(payload)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	expected := []float64{0.1, 0.2, 0.3}
	if len(embedding) != len(expected) {
		t.Fatalf("expected embedding length %d, got %d", len(expected), len(embedding))
	}
	for i, v := range expected {
		if embedding[i] != v {
			t.Errorf("expected embedding[%d]=%f, got %f", i, v, embedding[i])
		}
	}
}

// TestGetEmbedding_ClientError verifies that an error from the HTTP client is handled.
func TestGetEmbedding_ClientError(t *testing.T) {
	fakeClient := &FakeClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return nil, errors.New("client error")
		},
	}

	client := NewOpenAI("http://example.com", "test-key")
	client.client = fakeClient

	payload := GetEmbeddingPayload{
		Model: "test-model",
		Input: "Hello",
	}

	_, err := client.GetEmbedding(payload)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "client error") {
		t.Errorf("expected error to contain 'client error', got %v", err)
	}
}

// TestGetCompletion_Success verifies a simple completion without tool calls.
func TestGetCompletion_Success(t *testing.T) {
	// Create a fake completion response (no tool calls).
	completionMessage := LLMMessage{
		Role:    "assistant",
		Content: "Hello world",
	}
	completionResponse := CompletionResponse{
		Choices: []LLMChoice{
			{
				Index:   0,
				Message: &completionMessage,
			},
		},
		Usage: &LLMUsage{
			PromptTokens:     5,
			CompletionTokens: 10,
			TotalTokens:      15,
		},
	}
	respBody, _ := json.Marshal(completionResponse)
	fakeClient := &FakeClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return fakeResponse(200, string(respBody)), nil
		},
	}

	client := NewOpenAI("http://example.com", "test-key")
	client.client = fakeClient

	payload := &CompletionRequestPayload{
		Model:    "test-model",
		Messages: []LLMMessage{{Role: "user", Content: "Hi"}},
	}

	result, err := client.GetCompletion(payload)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if result != "Hello world" {
		t.Errorf("expected 'Hello world', got '%s'", result)
	}
}

// TestGetCompletion_WithToolCalls simulates a conversation that involves a tool call.
func TestGetCompletion_WithToolCalls(t *testing.T) {
	// First response: the assistant message contains a tool call.
	toolCall := ToolCall{
		Id:   "tool1",
		Type: "function",
		Function: FunctionCall{
			Name:      "echo",
			Arguments: "test argument",
		},
	}
	messageWithTool := LLMMessage{
		Role:      "assistant",
		ToolCalls: []ToolCall{toolCall},
	}
	completionResponse1 := CompletionResponse{
		Choices: []LLMChoice{
			{
				Index:   0,
				Message: &messageWithTool,
			},
		},
		Usage: &LLMUsage{},
	}
	respBody1, _ := json.Marshal(completionResponse1)

	// Second response: final assistant message with content and no tool calls.
	finalMessage := LLMMessage{
		Role:    "assistant",
		Content: "Final answer",
	}
	completionResponse2 := CompletionResponse{
		Choices: []LLMChoice{
			{
				Index:   0,
				Message: &finalMessage,
			},
		},
		Usage: &LLMUsage{},
	}
	respBody2, _ := json.Marshal(completionResponse2)

	// Set up a sequential fake client that returns the above responses.
	seqClient := &SequentialFakeClient{
		Responses: []*http.Response{
			fakeResponse(200, string(respBody1)),
			fakeResponse(200, string(respBody2)),
		},
	}

	client := NewOpenAI("http://example.com", "test-key")
	client.client = seqClient

	// Define a tool named "echo" whose function simply returns a modified string.
	echoFn := func(args string) string {
		return "echo: " + args
	}
	toolDef := NewToolDefinition(&FunctionDefinition{
		Name: "echo",
		Fn:   echoFn,
	})

	payload := &CompletionRequestPayload{
		Model:    "test-model",
		Messages: []LLMMessage{{Role: "user", Content: "Hi"}},
		Tools:    []ToolDefinition{toolDef},
	}

	result, err := client.GetCompletion(payload)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if result != "Final answer" {
		t.Errorf("expected 'Final answer', got '%s'", result)
	}

	// Verify that the tool call was handled: we expect the payload.Messages to now contain:
	// 1. The original user message
	// 2. The first assistant message (with tool call)
	// 3. A tool response message (added by handleToolCalls)
	// 4. The final assistant message.
	if len(payload.Messages) != 4 {
		t.Errorf("expected 4 messages in the payload, got %d", len(payload.Messages))
	}
	toolResponse := payload.Messages[2]
	if toolResponse.Role != "tool" {
		t.Errorf("expected tool response role to be 'tool', got '%s'", toolResponse.Role)
	}
	if toolResponse.Content != "echo: test argument" {
		t.Errorf("expected tool response content 'echo: test argument', got '%s'", toolResponse.Content)
	}
}

// TestGetCompletion_OpenAiRequestError verifies that an error in openAiRequest is propagated.
func TestGetCompletion_OpenAiRequestError(t *testing.T) {
	fakeClient := &FakeClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return nil, errors.New("openai request failed")
		},
	}

	client := NewOpenAI("http://example.com", "test-key")
	client.client = fakeClient

	payload := &CompletionRequestPayload{
		Model:    "test-model",
		Messages: []LLMMessage{{Role: "user", Content: "Hi"}},
	}

	_, err := client.GetCompletion(payload)
	if err == nil {
		t.Fatalf("expected an error, got nil")
	}
	if !strings.Contains(err.Error(), "openai request failed") {
		t.Errorf("expected error message to contain 'openai request failed', got '%v'", err)
	}
}

// TestCreateAuthorizedRequest verifies that the request is correctly created with the required headers.
func TestCreateAuthorizedRequest(t *testing.T) {
	req, err := createAuthorizedRequest("GET", "http://example.com/test", nil, "test-key")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got '%s'", req.Header.Get("Content-Type"))
	}
	authHeader := req.Header.Get("Authorization")
	if authHeader != "Bearer test-key" {
		t.Errorf("expected Authorization header 'Bearer test-key', got '%s'", authHeader)
	}
}
