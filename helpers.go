package openaiclient

import (
	"fmt"
	"io"
	"net/http"
)

func createRequest(method, endpoint string, body io.Reader) (*http.Request, error) {
	request, err := http.NewRequest(method, endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	request.Header.Add("Content-Type", "application/json")

	return request, nil
}

func createAuthorizedRequest(method, endpoint string, body io.Reader, bearer string) (*http.Request, error) {
	request, err := createRequest(method, endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("error creating authorized request: %w", err)
	}
	request.Header.Add("Authorization", fmt.Sprintf("Bearer %s", bearer))

	return request, nil
}
