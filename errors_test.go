package openaiclient

import (
	"errors"
	"net/http"
	"testing"
)

func TestNewOpenAIError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       []byte
		wantType   string
		wantMsg    string
		wantCode   string
	}{
		{
			name:       "invalid request error",
			statusCode: http.StatusBadRequest,
			body: []byte(`{
				"type": "invalid_request_error",
				"message": "Invalid model",
				"code": "invalid_model",
				"param": "model"
			}`),
			wantType: ErrTypeInvalidRequest,
			wantMsg:  "Invalid model",
			wantCode: "invalid_model",
		},
		{
			name:       "authentication error",
			statusCode: http.StatusUnauthorized,
			body: []byte(`{
				"type": "authentication_error",
				"message": "Invalid API key"
			}`),
			wantType: ErrTypeAuthentication,
			wantMsg:  "Invalid API key",
		},
		{
			name:       "rate limit error",
			statusCode: http.StatusTooManyRequests,
			body: []byte(`{
				"type": "rate_limit_error",
				"message": "Rate limit exceeded"
			}`),
			wantType: ErrTypeRateLimit,
			wantMsg:  "Rate limit exceeded",
		},
		{
			name:       "service unavailable error",
			statusCode: http.StatusServiceUnavailable,
			body: []byte(`{
				"type": "service_unavailable",
				"message": "Service is temporarily unavailable"
			}`),
			wantType: ErrTypeServiceUnavailable,
			wantMsg:  "Service is temporarily unavailable",
		},
		{
			name:       "not found error",
			statusCode: http.StatusNotFound,
			body: []byte(`{
				"type": "not_found",
				"message": "Resource not found"
			}`),
			wantType: ErrTypeNotFound,
			wantMsg:  "Resource not found",
		},
		{
			name:       "unknown error type",
			statusCode: http.StatusInternalServerError,
			body: []byte(`{
				"type": "unknown_error",
				"message": "Something went wrong"
			}`),
			wantType: "unknown_error",
			wantMsg:  "Something went wrong",
		},
		{
			name:       "invalid JSON response",
			statusCode: http.StatusBadRequest,
			body:       []byte(`invalid json`),
			wantType:   ErrTypeInvalidRequest,
			wantMsg:    "request failed with status 400: invalid json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewOpenAIError(tt.statusCode, tt.body)
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			if tt.name == "invalid JSON response" {
				if err.Error() != tt.wantMsg {
					t.Errorf("got error message %q, want %q", err.Error(), tt.wantMsg)
				}
				return
			}

			apiErr, ok := err.(*OpenAIError)
			if !ok {
				t.Fatalf("expected *OpenAIError, got %T", err)
			}

			if apiErr.Type != tt.wantType {
				t.Errorf("got type %q, want %q", apiErr.Type, tt.wantType)
			}

			if apiErr.Message != tt.wantMsg {
				t.Errorf("got message %q, want %q", apiErr.Message, tt.wantMsg)
			}

			if tt.wantCode != "" && apiErr.Code != tt.wantCode {
				t.Errorf("got code %q, want %q", apiErr.Code, tt.wantCode)
			}
		})
	}
}

func TestErrorConstructors(t *testing.T) {
	tests := []struct {
		name        string
		constructor func(string) error
		wantType    string
		message     string
	}{
		{
			name:        "invalid request error",
			constructor: NewInvalidRequestError,
			wantType:    ErrTypeInvalidRequest,
			message:     "Invalid request",
		},
		{
			name:        "authentication error",
			constructor: NewAuthenticationError,
			wantType:    ErrTypeAuthentication,
			message:     "Invalid credentials",
		},
		{
			name:        "rate limit error",
			constructor: NewRateLimitError,
			wantType:    ErrTypeRateLimit,
			message:     "Rate limit exceeded",
		},
		{
			name:        "service unavailable error",
			constructor: NewServiceUnavailableError,
			wantType:    ErrTypeServiceUnavailable,
			message:     "Service unavailable",
		},
		{
			name:        "not found error",
			constructor: NewNotFoundError,
			wantType:    ErrTypeNotFound,
			message:     "Resource not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.constructor(tt.message)
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			apiErr, ok := err.(*OpenAIError)
			if !ok {
				t.Fatalf("expected *OpenAIError, got %T", err)
			}

			if apiErr.Type != tt.wantType {
				t.Errorf("got type %q, want %q", apiErr.Type, tt.wantType)
			}

			if apiErr.Message != tt.message {
				t.Errorf("got message %q, want %q", apiErr.Message, tt.message)
			}
		})
	}
}

func TestIsOpenAIError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "OpenAI error",
			err:  &OpenAIError{Type: ErrTypeInvalidRequest, Message: "test"},
			want: true,
		},
		{
			name: "standard error",
			err:  errors.New("test error"),
			want: false,
		},
		{
			name: "nil error",
			err:  nil,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsOpenAIError(tt.err)
			if got != tt.want {
				t.Errorf("IsOpenAIError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetOpenAIErrorType(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "OpenAI error",
			err:  &OpenAIError{Type: ErrTypeInvalidRequest, Message: "test"},
			want: ErrTypeInvalidRequest,
		},
		{
			name: "standard error",
			err:  errors.New("test error"),
			want: "",
		},
		{
			name: "nil error",
			err:  nil,
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetOpenAIErrorType(tt.err)
			if got != tt.want {
				t.Errorf("GetOpenAIErrorType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNewOpenAIError_StatusCodeMapping(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       []byte
		wantType   string
	}{
		{
			name:       "bad request without type",
			statusCode: http.StatusBadRequest,
			body:       []byte(`{"message": "Bad request"}`),
			wantType:   ErrTypeInvalidRequest,
		},
		{
			name:       "unauthorized without type",
			statusCode: http.StatusUnauthorized,
			body:       []byte(`{"message": "Unauthorized"}`),
			wantType:   ErrTypeAuthentication,
		},
		{
			name:       "rate limit without type",
			statusCode: http.StatusTooManyRequests,
			body:       []byte(`{"message": "Rate limit exceeded"}`),
			wantType:   ErrTypeRateLimit,
		},
		{
			name:       "service unavailable without type",
			statusCode: http.StatusServiceUnavailable,
			body:       []byte(`{"message": "Service unavailable"}`),
			wantType:   ErrTypeServiceUnavailable,
		},
		{
			name:       "not found without type",
			statusCode: http.StatusNotFound,
			body:       []byte(`{"message": "Not found"}`),
			wantType:   ErrTypeNotFound,
		},
		{
			name:       "unknown status code",
			statusCode: http.StatusInternalServerError,
			body:       []byte(`{"message": "Internal server error"}`),
			wantType:   "unknown_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewOpenAIError(tt.statusCode, tt.body)
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			apiErr, ok := err.(*OpenAIError)
			if !ok {
				t.Fatalf("expected *OpenAIError, got %T", err)
			}

			if apiErr.Type != tt.wantType {
				t.Errorf("got type %q, want %q", apiErr.Type, tt.wantType)
			}
		})
	}
}
