package openaiclient

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

const (
	completionsEndpont = "/v1/chat/completions"
	embeddingsEndpoint = "/v1/embeddings"
)

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type OpenAI struct {
	baseUrl       string
	client        httpClient
	key           string
	MaxIterations int
}

func NewOpenAI(baseUrl, apiKey string) *OpenAI {
	if baseUrl == "" {
		baseUrl = os.Getenv("OPENAI_BASE_URL")
	}
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	return &OpenAI{
		baseUrl:       baseUrl,
		client:        &http.Client{},
		key:           apiKey,
		MaxIterations: 5,
	}
}

func NewDefaultOpenAI() *OpenAI {
	return NewOpenAI("", "")
}

func (o *OpenAI) GetCompletion(payload *CompletionRequestPayload) (*Message, error) {
	if payload.Model == "" {
		payload.Model = os.Getenv("OPENAI_MODEL")
	}
	return o.performReActLoop(payload, o.MaxIterations)
}

func (o *OpenAI) GetEmbedding(payload GetEmbeddingPayload) ([]float64, error) {
	request, err := o.createAuthorizedRequest(
		http.MethodPost,
		embeddingsEndpoint,
		payload,
	)
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

func (o *OpenAI) createAuthorizedRequest(method, endpoint string, body any) (*http.Request, error) {
	return createAuthorizedRequest(method, o.endpoint(endpoint), body, o.key)
}

func (o *OpenAI) performReActLoop(payload *CompletionRequestPayload, maxIterations int) (*Message, error) {
	for range maxIterations {
		if err := o.getCompletion(payload); err != nil {
			return nil, err
		}

		responseBody := payload.Messages[len(payload.Messages)-1]

		if len(responseBody.ToolCalls) == 0 {
			content := responseBody.Content
			if content != "" {
				log.Println("final response:")
				log.Println(content)
			}
			return &responseBody, nil
		}

		if err := o.handleToolCalls(payload); err != nil {
			return nil, fmt.Errorf("error handling tool calls: %w", err)
		}
	}

	return nil, fmt.Errorf("reached max iterations without finalizing an answer")
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

		payload.AddMessages(Message{
			Role:       MessageRoleTool,
			Content:    result,
			ToolCallId: toolCall.Id,
		})
	}
	return nil
}

func (o *OpenAI) getCompletion(payload *CompletionRequestPayload) error {
	request, err := o.createAuthorizedRequest(
		http.MethodPost,
		completionsEndpont,
		payload,
	)
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

	if len(responseBody.Choices) == 0 {
		return fmt.Errorf("no choices returned")
	}

	payload.AddMessages(*responseBody.Choices[0].Message)

	return nil
}
