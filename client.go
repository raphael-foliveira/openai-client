package openaiclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type OpenAI struct {
	baseUrl string
	client  httpClient
	key     string
}

func NewOpenAI(baseUrl, apiKey string) *OpenAI {
	if baseUrl == "" {
		baseUrl = os.Getenv("OPENAI_BASE_URL")
	}
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	return &OpenAI{
		baseUrl: baseUrl,
		client:  &http.Client{},
		key:     apiKey,
	}
}

func NewDefaultOpenAI() *OpenAI {
	return NewOpenAI("", "")
}

func (o *OpenAI) GetCompletion(payload *CompletionRequestPayload) (string, error) {
	if payload.Model == "" {
		payload.Model = os.Getenv("OPENAI_MODEL")
	}

	content, err := o.performReActLoop(payload, 5)
	if err != nil {
		return content, err
	}

	return content, nil
}

func (o *OpenAI) GetEmbedding(payload GetEmbeddingPayload) ([]float64, error) {
	requestBody, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request payload: %w", err)
	}

	request, err := o.createAuthorizedRequest("POST", "/v1/embeddings", bytes.NewReader(requestBody))
	if err != nil {
		return nil, err
	}

	response, err := o.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer response.Body.Close()

	var responseBody GetEmbeddingResponse
	if err := json.NewDecoder(response.Body).Decode(&responseBody); err != nil {
		return nil, fmt.Errorf("error unmarshaling response body: %w", err)
	}

	return responseBody.Data[0].Embedding, nil
}

func (o *OpenAI) endpoint(e string) string {
	return fmt.Sprintf("%s%s", o.baseUrl, e)
}

func (o *OpenAI) createAuthorizedRequest(method, endpoint string, body io.Reader) (*http.Request, error) {
	return createAuthorizedRequest(method, o.endpoint(endpoint), body, o.key)
}

func (o *OpenAI) performReActLoop(payload *CompletionRequestPayload, maxIterations int) (string, error) {
	for range maxIterations {
		if err := o.openAiRequest(payload); err != nil {
			return "", err
		}

		responseBody := payload.Messages[len(payload.Messages)-1]

		if len(responseBody.ToolCalls) == 0 {
			content := responseBody.Content
			if content != "" {
				log.Println("final response:")
				log.Println(content)
			}
			return content, nil
		}

		if err := o.handleToolCalls(payload); err != nil {
			return "", fmt.Errorf("error handling tool calls: %w", err)
		}
	}

	return "", fmt.Errorf("reached max iterations without finalizing an answer")
}

func (o *OpenAI) handleToolCalls(payload *CompletionRequestPayload) error {
	log.Println("handling tool calls")

	message := payload.Messages[len(payload.Messages)-1]

	for _, toolCall := range message.ToolCalls {
		fnName := toolCall.Function.Name
		arguments := toolCall.Function.Arguments
		tool, toolFound := payload.toolsMap()[fnName]
		if !toolFound {
			log.Println("tool not found:", fnName)
			continue
		}

		log.Println("calling tool:", fnName)

		result := tool.Fn(arguments)

		payload.Messages = append(payload.Messages, LLMMessage{
			Role:       "tool",
			Content:    result,
			ToolCallId: toolCall.Id,
		})
	}
	return nil
}

func (o *OpenAI) openAiRequest(payload *CompletionRequestPayload) error {
	requestBody, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("error marshaling request payload: %w", err)
	}

	request, err := o.createAuthorizedRequest("POST", "/v1/chat/completions", bytes.NewReader(requestBody))
	if err != nil {
		return err
	}

	response, err := o.client.Do(request)
	if err != nil {
		return fmt.Errorf("error making request: %w", err)
	}
	defer response.Body.Close()

	responseText, err := io.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	var responseBody CompletionResponse

	if err := json.Unmarshal(responseText, &responseBody); err != nil {
		return fmt.Errorf("error unmarshaling response body: %w", err)
	}

	payload.Messages = append(payload.Messages, *responseBody.Choices[0].Message)

	return nil
}
