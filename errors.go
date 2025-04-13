package openaiclient

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

const (
	ErrTypeInvalidRequest     = "invalid_request_error"
	ErrTypeAuthentication     = "authentication_error"
	ErrTypeRateLimit          = "rate_limit_error"
	ErrTypeServiceUnavailable = "service_unavailable"
	ErrTypeNotFound           = "not_found"
)

type OpenAIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
}

func (e *OpenAIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s (code: %s)", e.Type, e.Message, e.Code)
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

func NewOpenAIError(statusCode int, body []byte) error {
	var apiErr OpenAIError
	if err := json.Unmarshal(body, &apiErr); err != nil {
		return fmt.Errorf("request failed with status %d: %s", statusCode, string(body))
	}

	if apiErr.Type == "" {
		switch statusCode {
		case http.StatusBadRequest:
			apiErr.Type = ErrTypeInvalidRequest
		case http.StatusUnauthorized:
			apiErr.Type = ErrTypeAuthentication
		case http.StatusTooManyRequests:
			apiErr.Type = ErrTypeRateLimit
		case http.StatusServiceUnavailable:
			apiErr.Type = ErrTypeServiceUnavailable
		case http.StatusNotFound:
			apiErr.Type = ErrTypeNotFound
		default:
			apiErr.Type = "unknown_error"
		}
	}

	return &apiErr
}

func NewInvalidRequestError(message string) error {
	return &OpenAIError{
		Type:    ErrTypeInvalidRequest,
		Message: message,
	}
}

func NewAuthenticationError(message string) error {
	return &OpenAIError{
		Type:    ErrTypeAuthentication,
		Message: message,
	}
}

func NewRateLimitError(message string) error {
	return &OpenAIError{
		Type:    ErrTypeRateLimit,
		Message: message,
	}
}

func NewServiceUnavailableError(message string) error {
	return &OpenAIError{
		Type:    ErrTypeServiceUnavailable,
		Message: message,
	}
}

func NewNotFoundError(message string) error {
	return &OpenAIError{
		Type:    ErrTypeNotFound,
		Message: message,
	}
}

func IsOpenAIError(err error) bool {
	var apiErr *OpenAIError
	return errors.As(err, &apiErr)
}

func GetOpenAIErrorType(err error) string {
	if apiErr, ok := err.(*OpenAIError); ok {
		return apiErr.Type
	}
	return ""
}
