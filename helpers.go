package openaiclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

func createRequest(method, endpoint string, body any) (*http.Request, error) {
	bodyJson, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request body: %w", err)
	}
	request, err := http.NewRequest(method, endpoint, bytes.NewReader(bodyJson))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	request.Header.Add("Content-Type", "application/json")

	return request, nil
}

func createAuthorizedRequest(method, endpoint string, body any, bearer string) (*http.Request, error) {
	request, err := createRequest(method, endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("error creating authorized request: %w", err)
	}
	request.Header.Add("Authorization", fmt.Sprintf("Bearer %s", bearer))

	return request, nil
}
