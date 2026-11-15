package config

import "testing"

func TestServerValidateSecurity(t *testing.T) {
	t.Setenv("NAVIGATOR_AUTH_TOKEN", "")
	if err := (ServerConfig{}).ValidateSecurity(); err == nil {
		t.Error("no token and no opt-in must be refused")
	}
	if err := (ServerConfig{AuthToken: "   "}).ValidateSecurity(); err == nil {
		t.Error("a blank token must be refused")
	}
	if err := (ServerConfig{AuthToken: "s3cret"}).ValidateSecurity(); err != nil {
		t.Errorf("configured token should be accepted: %v", err)
	}
	if err := (ServerConfig{AllowUnauthenticated: true}).ValidateSecurity(); err != nil {
		t.Errorf("explicit dev opt-in should be accepted: %v", err)
	}
	t.Setenv("NAVIGATOR_AUTH_TOKEN", "from-env")
	if err := (ServerConfig{}).ValidateSecurity(); err != nil {
		t.Errorf("env token should be accepted: %v", err)
	}
}
