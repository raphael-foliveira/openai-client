package main

import (
	"encoding/json"
	"fmt"
	"log"

	openaiclient "github.com/raphael-foliveira/openai-client"
)

func main() {
	c := openaiclient.NewDefault()

	payload := &openaiclient.CompletionRequestPayload{
		Messages: []openaiclient.Message{
			{
				Role:    openaiclient.MessageRoleSystem,
				Content: "You are a colorful pirate. You must include as many colors as possible in your responses.",
			},
			{
				Role:    openaiclient.MessageRoleUser,
				Content: "Hello, how are you?",
			},
		},
	}

	response, err := c.GetCompletion(payload)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Println("response:", prettySprint(response))
}

func prettySprint(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	return string(b)
}
