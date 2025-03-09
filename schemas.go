package openaiclient

type MessageRole string

var (
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleSystem    MessageRole = "system"
	MessageRoleDeveloper MessageRole = "developer"
	MessageRoleTool      MessageRole = "tool"
)

type (
	JsonSchemaProperties map[string]*JsonSchema
	JsonSchema           struct {
		Type        string               `json:"type,omitempty"`
		Description string               `json:"description,omitempty"`
		Properties  JsonSchemaProperties `json:"properties,omitempty"`
		Required    []string             `json:"required,omitempty"`
		Items       *JsonSchema          `json:"items,omitempty"`
	}

	ToolResult struct {
		Error  string `json:"error,omitempty"`
		Result string `json:"result,omitempty"`
	}

	LLMTool = func(string) string

	FunctionCall struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}

	ToolCall struct {
		Id       string       `json:"id"`
		Type     string       `json:"type"`
		Function FunctionCall `json:"function"`
	}

	FunctionDefinition struct {
		Name        string      `json:"name"`
		Description string      `json:"description,omitempty"`
		Parameters  *JsonSchema `json:"parameters,omitempty"`
		Fn          LLMTool     `json:"-"`
	}

	ToolDefinition struct {
		Type     string              `json:"type"` // "function"
		Function *FunctionDefinition `json:"function"`
	}

	Message struct {
		Role       MessageRole `json:"role"`
		Content    string      `json:"content"`
		ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
		Name       string      `json:"name,omitempty"`
		ToolCallId string      `json:"tool_call_id,omitempty"`
	}

	GetEmbeddingPayload struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}

	CompletionRequestPayload struct {
		Model       string           `json:"model,omitempty"`
		Messages    []Message        `json:"messages"`
		NewMessages []Message        `json:"-"`
		Tools       []ToolDefinition `json:"tools,omitempty"`
		ToolChoice  any              `json:"tool_choice,omitempty"`
	}

	LLMUsage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}

	LLMChoice struct {
		Index   int      `json:"index"`
		Message *Message `json:"message"`
	}

	CompletionResponse struct {
		Choices []LLMChoice `json:"choices"`
		Usage   *LLMUsage   `json:"usage"`
	}

	EmbeddingObject struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
		Model     string    `json:"model"`
	}

	GetEmbeddingResponse struct {
		Object string            `json:"object"`
		Data   []EmbeddingObject `json:"data"`
		Usage  *LLMUsage         `json:"usage"`
	}
)

func (c *CompletionRequestPayload) toolsMap() map[string]*FunctionDefinition {
	toolsMap := make(map[string]*FunctionDefinition)
	for _, tool := range c.Tools {
		toolsMap[tool.Function.Name] = tool.Function
	}
	return toolsMap
}

// NewToolDefinition creates a new ToolDefinition with the given FunctionDefinition
// by automatically populating the "Type" field with "function".
func NewToolDefinition(functionDefinition *FunctionDefinition) ToolDefinition {
	return ToolDefinition{
		Type:     "function",
		Function: functionDefinition,
	}
}

func (c *CompletionRequestPayload) AddMessages(messages ...Message) {
	c.Messages = append(c.Messages, messages...)
	c.NewMessages = append(c.NewMessages, messages...)
}
