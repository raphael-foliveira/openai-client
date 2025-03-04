package openaiclient

import (
	"log"
	"strings"
	"testing"
)

var openAiClient *OpenAI

func TestMain(m *testing.M) {
	openAiClient = NewDefaultOpenAI()
	m.Run()
}

func TestOpenAI(t *testing.T) {
	t.Run("Should return an llm response", func(t *testing.T) {
		expected := "test passed"
		payload := &CompletionRequestPayload{
			Messages: []LLMMessage{
				{Role: "system", Content: "You must include 'test passed' in your response"},
				{Role: "user", Content: "Hello, world!"},
			},
		}

		response, err := openAiClient.GetCompletion(payload)
		if err != nil {
			t.Fatalf("Error getting completion: %s", err)
		}

		if !strings.Contains(strings.ToLower(response), "test passed") {
			t.Errorf("Expected response to contain '%s', got '%s'", expected, response)
		}

		log.Println("response:", response)
	})
}
