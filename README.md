# OpenAI Client

A lightweight, idiomatic Go client for the OpenAI API.

## Features

- Simple and intuitive API
- Support for OpenAI's chat completions API
- Support for embeddings API
- Configurable retry mechanism
- Environment variable configuration
- Tool/function calling support

## Installation

```bash
go get github.com/yourusername/openai-client
```

## Quick Start

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/yourusername/openai-client"
)

func main() {
	// Create a new client
	client, err := openaiclient.New("", "")
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	// Create a completion request
	payload := &openaiclient.CompletionRequestPayload{
		Model: "gpt-4",
		Messages: []openaiclient.Message{
			{
				Role:    openaiclient.MessageRoleUser,
				Content: "Hello, how are you?",
			},
		},
	}

	// Get a completion
	response, err := client.GetCompletion(payload)
	if err != nil {
		log.Fatalf("Failed to get completion: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Content)
}
```

## Configuration

The client can be configured using environment variables or directly when creating a new client:

```go
// Using environment variables
client, err := openaiclient.NewDefault()

// Using direct configuration
client, err := openaiclient.New("https://api.openai.com", "your-api-key")
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: The base URL for the OpenAI API (defaults to "https://api.openai.com")
- `OPENAI_MODEL`: The default model to use for completions (defaults to "gpt-4o-mini")

## Advanced Usage

### Tool/Function Calling

```go
// Define a tool
echoTool := openaiclient.NewToolDefinition(&openaiclient.FunctionDefinition{
	Name: "echo",
	Fn: func(args string) string {
		return "Echo: " + args
	},
})

// Create a completion request with tools
payload := &openaiclient.CompletionRequestPayload{
	Model: "gpt-4",
	Messages: []openaiclient.Message{
		{
			Role:    openaiclient.MessageRoleUser,
			Content: "Use the echo tool to repeat 'Hello, world!'",
		},
	},
	Tools: []openaiclient.ToolDefinition{echoTool},
}

// Get a completion
response, err := client.GetCompletion(payload)
```

### Embeddings

```go
// Create an embedding request
payload := openaiclient.GetEmbeddingPayload{
	Model: "text-embedding-3-small",
	Input: "Hello, world!",
}

// Get an embedding
embedding, err := client.GetEmbedding(payload)
if err != nil {
	log.Fatalf("Failed to get embedding: %v", err)
}

fmt.Printf("Embedding: %v\n", embedding)
```
