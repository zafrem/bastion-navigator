package vault

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

func TestVaultNavigatorCommunication(t *testing.T) {
	// 1. Start Vault Server in its own process group so all descendants are killed on
	// teardown. "go run" compiles and then forks a child process; killing only the
	// "go run" PID leaves that child orphaned, causing the "I/O incomplete" failure.
	vaultCmd := exec.Command("go", "run", "../../../vault/cmd/vault/main.go", "server", "--config", "../../../tests/configs/vault.yaml")

	vaultCmd.Stdout = os.Stdout
	vaultCmd.Stderr = os.Stderr
	vaultCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	err := vaultCmd.Start()
	if err != nil {
		t.Fatalf("Failed to start Vault: %v", err)
	}
	defer func() {
		syscall.Kill(-vaultCmd.Process.Pid, syscall.SIGKILL)
		vaultCmd.Wait()
	}()

	// Wait for Vault to be ready
	time.Sleep(10 * time.Second)

	// 2. Test Navigator's Vault Client
	// Using the local NewHTTP since we are in the same package
	vlt := NewHTTP("http://localhost:8081", nil, 1*time.Minute)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	fmt.Println(">>> Navigator calling Vault /v1/vault/permissions/user123...")
	allowed, err := vlt.AllowedCategories(ctx, "user123")
	if err != nil {
		t.Fatalf("Navigator failed to communicate with Vault: %v", err)
	}

	fmt.Printf(">>> Vault responded with allowed categories: %v\n", allowed)

	if len(allowed) == 0 {
		t.Errorf("Expected some allowed categories, got none")
	}
}
