package swarmgo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/yuanxiangyx/swarmgo-plus/llm"
)

var (
	// Define common errors for better error handling
	ErrNilAgent          = errors.New("agent cannot be nil")
	ErrEmptyMessages     = errors.New("message history cannot be empty")
	ErrLLMClientNotReady = errors.New("LLM client is not initialized")
	ErrInvalidProvider   = errors.New("invalid LLM provider specified")
	ErrNoChoicesInResp   = errors.New("no choices in LLM response")
	ErrMessageTooLong    = errors.New("message exceeds maximum token limit")
)

// Swarm represents the main structure
type Swarm struct {
	client          llm.LLM
	clients         map[llm.LLMProvider]llm.LLM
	providerConfigs map[llm.LLMProvider]*ClientConfig
	defaultProvider llm.LLMProvider
	tokenCounter    func(string) int // Optional token counter function
	initialized     bool             // Flag to check if Swarm is properly initialized
	config          *Config          // Configuration settings
}

// Config holds configuration options for Swarm
type Config struct {
	MaxRetries        int
	RetryBackoff      time.Duration
	RequestTimeout    time.Duration
	MaxTokens         int
	DefaultModel      string
	Debug             bool
	LogLevel          LogLevel
	TokenLimits       map[string]int // Model-specific token limits
	FailureHandlers   []FailureHandler
	RateLimitStrategy RateLimitStrategy
}

// LogLevel represents the level of logging
type LogLevel int

const (
	LogSilent LogLevel = iota
	LogError
	LogWarning
	LogInfo
	LogDebug
	LogTrace
)

// FailureHandler defines a function to handle specific failures
type FailureHandler func(error) (bool, error)

// RateLimitStrategy defines how rate limits are handled
type RateLimitStrategy int

const (
	RateLimitRetry RateLimitStrategy = iota
	RateLimitFail
	RateLimitQueue
)

// DefaultConfig returns default configuration values
func DefaultConfig() *Config {
	return &Config{
		MaxRetries:     3,
		RetryBackoff:   time.Second,
		RequestTimeout: 60 * time.Second,
		MaxTokens:      4096,
		DefaultModel:   "gpt-3.5-turbo",
		Debug:          false,
		LogLevel:       LogError,
		TokenLimits: map[string]int{
			"gpt-3.5-turbo": 4096,
			"gpt-4":         8192,
			"gpt-4o":        128000,
			"claude-3-opus": 200000,
		},
		RateLimitStrategy: RateLimitRetry,
	}
}

func normalizeConfig(config *Config) *Config {
	if config == nil {
		return DefaultConfig()
	}
	return config
}

// NewSwarm initializes a new Swarm instance with an LLM client
func NewSwarm(apiKey string, provider llm.LLMProvider) *Swarm {
	return NewSwarmWithConfig(apiKey, provider, DefaultConfig())
}

// NewSwarmWithConfig initializes a new Swarm with custom configuration
func NewSwarmWithConfig(apiKey string, provider llm.LLMProvider, config *Config) *Swarm {
	sw := &Swarm{
		clients:         make(map[llm.LLMProvider]llm.LLM),
		providerConfigs: make(map[llm.LLMProvider]*ClientConfig),
		defaultProvider: provider,
		config:          normalizeConfig(config),
	}

	if provider == "" {
		log.Println("Warning: Empty provider provided")
		return sw
	}

	if apiKey == "" && providerRequiresAuth(provider) {
		log.Println("Warning: Empty API key provided")
		return sw
	}

	if err := sw.RegisterProvider(apiKey, provider); err != nil {
		log.Printf("Failed to initialize %s client: %v", provider, err)
		return sw
	}

	return sw
}

// NewSwarmWithHost creates a Swarm with a custom host
func NewSwarmWithHost(apiKey, host string, provider llm.LLMProvider) *Swarm {
	sw := &Swarm{
		clients:         make(map[llm.LLMProvider]llm.LLM),
		providerConfigs: make(map[llm.LLMProvider]*ClientConfig),
		defaultProvider: provider,
		config:          DefaultConfig(),
	}

	if err := sw.RegisterProviderConfig(provider, &ClientConfig{
		Provider:  provider,
		AuthToken: apiKey,
		BaseURL:   host,
	}); err != nil {
		log.Printf("Failed to initialize %s client with custom host: %v", provider, err)
		return sw
	}

	return sw
}

// NewSwarmWithCustomProvider creates a Swarm with a custom LLM provider implementation
func NewSwarmWithCustomProvider(providerImpl llm.LLM, config *Config) *Swarm {
	return &Swarm{
		client:          providerImpl,
		clients:         make(map[llm.LLMProvider]llm.LLM),
		providerConfigs: make(map[llm.LLMProvider]*ClientConfig),
		initialized:     providerImpl != nil,
		config:          normalizeConfig(config),
	}
}

// SetTokenCounter sets a function to count tokens in messages
func (s *Swarm) SetTokenCounter(counter func(string) int) {
	s.tokenCounter = counter
}

// IsInitialized returns whether the Swarm is properly initialized
func (s *Swarm) IsInitialized() bool {
	return s.initialized && (s.client != nil || len(s.clients) > 0)
}

// ValidateConnection tests the LLM connection with a simple request
func (s *Swarm) ValidateConnection(ctx context.Context) error {
	if !s.IsInitialized() {
		return ErrLLMClientNotReady
	}

	client, err := s.resolveClient(nil)
	if err != nil {
		return err
	}

	// Create a simple test request
	testRequest := llm.ChatCompletionRequest{
		Model: s.swarmConfig().DefaultModel,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Test connection"},
		},
		MaxTokens: 5,
	}

	// Attempt to send the request
	_, err = client.CreateChatCompletion(ctx, testRequest)
	if err != nil {
		return fmt.Errorf("connection test failed: %w", err)
	}

	return nil
}

// RegisterProvider registers or replaces a provider on the swarm instance.
func (s *Swarm) RegisterProvider(apiKey string, provider llm.LLMProvider) error {
	return s.RegisterProviderConfig(provider, &ClientConfig{
		Provider:  provider,
		AuthToken: apiKey,
	})
}

// RegisterProviderConfig registers or replaces a provider using a full client config.
func (s *Swarm) RegisterProviderConfig(provider llm.LLMProvider, config *ClientConfig) error {
	if provider == "" {
		return ErrInvalidProvider
	}

	if s.clients == nil {
		s.clients = make(map[llm.LLMProvider]llm.LLM)
	}
	if s.providerConfigs == nil {
		s.providerConfigs = make(map[llm.LLMProvider]*ClientConfig)
	}
	if s.config == nil {
		s.config = DefaultConfig()
	}

	mergedConfig := mergeClientConfig(nil, config, provider)
	client, err := newLLMClient(provider, mergedConfig)
	if err != nil {
		return err
	}
	if client == nil {
		return ErrLLMClientNotReady
	}

	s.clients[provider] = client
	s.providerConfigs[provider] = mergedConfig
	if s.defaultProvider == "" {
		s.defaultProvider = provider
	}
	if provider == s.defaultProvider || s.client == nil {
		s.client = client
	}
	s.initialized = true

	return nil
}

func (s *Swarm) swarmConfig() *Config {
	if s == nil || s.config == nil {
		return DefaultConfig()
	}
	return s.config
}

func providerRequiresAuth(provider llm.LLMProvider) bool {
	return provider != llm.Ollama
}

func newLLMClient(provider llm.LLMProvider, config *ClientConfig) (llm.LLM, error) {
	config = mergeClientConfig(nil, config, provider)

	if providerRequiresAuth(provider) && config.AuthToken == "" {
		return nil, fmt.Errorf("provider %s requires an auth token", provider)
	}

	switch provider {
	case llm.OpenAI:
		if config.BaseURL != "" {
			return llm.NewOpenAILLMWithHost(config.AuthToken, config.BaseURL), nil
		}
		return llm.NewOpenAILLM(config.AuthToken), nil
	case llm.Gemini:
		return llm.NewGeminiLLM(config.AuthToken)
	case llm.Claude:
		return llm.NewClaudeLLM(config.AuthToken), nil
	case llm.Ollama:
		if config.BaseURL != "" {
			return llm.NewOllamaLLMWithURL(config.BaseURL)
		}
		return llm.NewOllamaLLM()
	case llm.DeepSeek:
		if config.BaseURL != "" {
			return llm.NewDeepSeekLLMWithEndpoint(config.AuthToken, config.BaseURL), nil
		}
		return llm.NewDeepSeekLLM(config.AuthToken), nil
	default:
		return nil, fmt.Errorf("%w: %s", ErrInvalidProvider, provider)
	}
}

func cloneClientConfig(config *ClientConfig) *ClientConfig {
	if config == nil {
		return nil
	}

	cloned := *config
	if config.Options != nil {
		cloned.Options = make(map[string]interface{}, len(config.Options))
		for k, v := range config.Options {
			cloned.Options[k] = v
		}
	}

	return &cloned
}

func mergeClientConfig(base, override *ClientConfig, provider llm.LLMProvider) *ClientConfig {
	var merged ClientConfig
	if base != nil {
		clonedBase := cloneClientConfig(base)
		merged = *clonedBase
	}

	if merged.Provider == "" {
		merged.Provider = provider
	}

	if override == nil {
		if merged.Options == nil {
			merged.Options = make(map[string]interface{})
		}
		return &merged
	}

	if override.Provider != "" {
		merged.Provider = override.Provider
	}
	if override.AuthToken != "" {
		merged.AuthToken = override.AuthToken
	}
	if override.BaseURL != "" {
		merged.BaseURL = override.BaseURL
	}
	if override.OrgID != "" {
		merged.OrgID = override.OrgID
	}
	if override.APIVersion != "" {
		merged.APIVersion = override.APIVersion
	}
	if override.AssistantVersion != "" {
		merged.AssistantVersion = override.AssistantVersion
	}
	if override.ModelMapperFunc != nil {
		merged.ModelMapperFunc = override.ModelMapperFunc
	}
	if override.HTTPClient != nil {
		merged.HTTPClient = override.HTTPClient
	}
	if override.EmptyMessagesLimit != 0 {
		merged.EmptyMessagesLimit = override.EmptyMessagesLimit
	}
	if merged.Options == nil {
		merged.Options = make(map[string]interface{})
	}
	for k, v := range override.Options {
		merged.Options[k] = v
	}

	if merged.Provider == "" {
		merged.Provider = provider
	}

	return &merged
}

func clientConfigNeedsDedicatedClient(config *ClientConfig) bool {
	if config == nil {
		return false
	}

	return config.AuthToken != "" ||
		config.BaseURL != "" ||
		config.OrgID != "" ||
		config.APIVersion != "" ||
		config.AssistantVersion != "" ||
		config.HTTPClient != nil ||
		len(config.Options) > 0
}

func (s *Swarm) resolveProvider(agent *Agent) llm.LLMProvider {
	if agent != nil && agent.Config != nil && agent.Config.Provider != "" {
		return agent.Config.Provider
	}
	if agent != nil && agent.Provider != "" {
		return agent.Provider
	}
	return s.defaultProvider
}

func (s *Swarm) effectiveClientConfig(agent *Agent, provider llm.LLMProvider) *ClientConfig {
	var baseConfig *ClientConfig
	if s != nil && s.providerConfigs != nil {
		baseConfig = s.providerConfigs[provider]
	}
	if agent == nil {
		return mergeClientConfig(baseConfig, nil, provider)
	}
	return mergeClientConfig(baseConfig, agent.Config, provider)
}

func (s *Swarm) resolveClient(agent *Agent) (llm.LLM, error) {
	if s == nil {
		return nil, ErrLLMClientNotReady
	}

	provider := s.resolveProvider(agent)
	if provider == "" {
		if s.client != nil {
			return s.client, nil
		}
		return nil, ErrLLMClientNotReady
	}

	if agent != nil && clientConfigNeedsDedicatedClient(agent.Config) {
		return newLLMClient(provider, s.effectiveClientConfig(agent, provider))
	}

	if client, ok := s.clients[provider]; ok && client != nil {
		return client, nil
	}

	if provider == s.defaultProvider && s.client != nil {
		return s.client, nil
	}

	config := s.effectiveClientConfig(agent, provider)
	if config == nil || (providerRequiresAuth(provider) && config.AuthToken == "") {
		return nil, fmt.Errorf(
			"provider %s is not configured; register it on the swarm or set agent.Config.AuthToken",
			provider,
		)
	}

	client, err := newLLMClient(provider, config)
	if err != nil {
		return nil, err
	}

	if s.clients == nil {
		s.clients = make(map[llm.LLMProvider]llm.LLM)
	}
	s.clients[provider] = client
	if s.providerConfigs == nil {
		s.providerConfigs = make(map[llm.LLMProvider]*ClientConfig)
	}
	s.providerConfigs[provider] = config
	if s.client == nil || provider == s.defaultProvider {
		s.client = client
	}
	s.initialized = true

	return client, nil
}

func resolveInstructions(agent *Agent, contextVariables map[string]interface{}) string {
	if agent == nil {
		return ""
	}
	if agent.InstructionsFunc == nil {
		return agent.Instructions
	}
	if contextVariables == nil {
		contextVariables = make(map[string]interface{})
	}
	return agent.InstructionsFunc(contextVariables)
}

func withAgentInstructions(history []llm.Message, instructions string, replaceExisting bool) []llm.Message {
	updated := cloneMessages(history)
	if instructions == "" {
		return updated
	}

	if len(updated) > 0 && updated[0].Role == llm.RoleSystem {
		if replaceExisting {
			updated[0].Content = instructions
		}
		return updated
	}

	return append([]llm.Message{
		{
			Role:    llm.RoleSystem,
			Content: instructions,
		},
	}, updated...)
}

func (s *Swarm) resolveModel(agent *Agent, modelOverride string) string {
	model := ""
	if agent != nil {
		model = agent.Model
	}
	if modelOverride != "" {
		model = modelOverride
	}
	if model == "" {
		model = s.swarmConfig().DefaultModel
	}

	provider := s.resolveProvider(agent)
	config := s.effectiveClientConfig(agent, provider)
	if config != nil && config.ModelMapperFunc != nil {
		model = config.ModelMapperFunc(model)
	}

	return model
}

func buildToolsForAgent(agent *Agent) []llm.Tool {
	if agent == nil || agent.Functions == nil {
		return nil
	}

	tools := make([]llm.Tool, 0, len(agent.Functions))
	for _, af := range agent.Functions {
		def := FunctionToDefinition(af)
		tools = append(tools, llm.Tool{
			Type: "function",
			Function: &llm.Function{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.Parameters,
			},
		})
	}

	return tools
}

// getChatCompletion requests a chat completion from the LLM with retries and error handling
func (s *Swarm) getChatCompletion(
	ctx context.Context,
	agent *Agent,
	history []llm.Message,
	contextVariables map[string]interface{},
	modelOverride string,
	stream bool,
	debug bool,
) (llm.ChatCompletionResponse, error) {
	// Validate inputs
	if agent == nil {
		return llm.ChatCompletionResponse{}, ErrNilAgent
	}

	if len(history) == 0 {
		// Instead of failing, create an empty initial message
		history = []llm.Message{}
	}

	history = withAgentInstructions(history, resolveInstructions(agent, contextVariables), false)

	model := s.resolveModel(agent, modelOverride)

	req := llm.ChatCompletionRequest{
		Model:    model,
		Messages: history,
		Tools:    buildToolsForAgent(agent),
	}

	if debug {
		log.Printf("Debug - Model: %s, Messages: %d, Tools: %d\n",
			model, len(history), len(req.Tools))
	}

	client, err := s.resolveClient(agent)
	if err != nil {
		return llm.ChatCompletionResponse{}, err
	}

	cfg := s.swarmConfig()

	// Implement retry logic
	var lastErr error
	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		if attempt > 0 && cfg.Debug {
			log.Printf("Retry attempt %d after error: %v", attempt, lastErr)
		}

		// Create a timeout context for this request if none was provided
		requestCtx := ctx
		if _, hasDeadline := ctx.Deadline(); !hasDeadline {
			var cancel context.CancelFunc
			requestCtx, cancel = context.WithTimeout(ctx, cfg.RequestTimeout)
			defer cancel()
		}

		// Call the LLM to get a chat completion
		resp, err := client.CreateChatCompletion(requestCtx, req)
		if err == nil {
			// Success
			return resp, nil
		}

		// Handle the error
		lastErr = err

		// Check for rate limit errors and apply the rate limit strategy
		if isRateLimitError(err) {
			switch cfg.RateLimitStrategy {
			case RateLimitFail:
				return llm.ChatCompletionResponse{}, fmt.Errorf("rate limit exceeded: %w", err)
			case RateLimitQueue:
				// Implement exponential backoff
				backoff := cfg.RetryBackoff * time.Duration(1<<uint(attempt))
				if cfg.Debug {
					log.Printf("Rate limit hit, backing off for %v", backoff)
				}
				select {
				case <-ctx.Done():
					return llm.ChatCompletionResponse{}, ctx.Err()
				case <-time.After(backoff):
					// Continue to next retry
				}
			default: // RateLimitRetry
				// Simple retry with backoff
				backoff := cfg.RetryBackoff * time.Duration(attempt+1)
				if cfg.Debug {
					log.Printf("Backing off for %v before retry", backoff)
				}
				time.Sleep(backoff)
			}
		} else if isFatalError(err) {
			// Don't retry fatal errors
			return llm.ChatCompletionResponse{}, err
		} else {
			// For other errors, apply backoff
			backoff := cfg.RetryBackoff * time.Duration(attempt+1)
			time.Sleep(backoff)
		}
	}

	// All retries failed
	return llm.ChatCompletionResponse{}, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// isRateLimitError checks if an error is related to rate limiting
func isRateLimitError(err error) bool {
	return err != nil && (strings.Contains(strings.ToLower(err.Error()), "rate limit") ||
		strings.Contains(strings.ToLower(err.Error()), "too many requests") ||
		strings.Contains(strings.ToLower(err.Error()), "429"))
}

// isFatalError checks if an error is fatal and should not be retried
func isFatalError(err error) bool {
	return err != nil && (strings.Contains(strings.ToLower(err.Error()), "invalid auth") ||
		strings.Contains(strings.ToLower(err.Error()), "authentication") ||
		strings.Contains(strings.ToLower(err.Error()), "not found") ||
		strings.Contains(strings.ToLower(err.Error()), "invalid model"))
}

// Helper function to clone a slice of messages
func cloneMessages(msgs []llm.Message) []llm.Message {
	if msgs == nil {
		return []llm.Message{}
	}

	cloned := make([]llm.Message, len(msgs))
	copy(cloned, msgs)
	return cloned
}

// Helper function to return the last message in a slice
func lastMessage(msgs []llm.Message) *llm.Message {
	if len(msgs) == 0 {
		return nil
	}
	return &msgs[len(msgs)-1]
}

// handleToolCall processes a tool call and ensures proper context
func (s *Swarm) handleToolCall(
	ctx context.Context,
	toolCall *llm.ToolCall,
	agent *Agent,
	contextVariables map[string]interface{},
	debug bool,
) (Response, error) {
	toolName := toolCall.Function.Name
	argsJSON := toolCall.Function.Arguments

	// Parse the tool call arguments
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		errorMsg := fmt.Sprintf("Error parsing tool call arguments: %v", err)
		if debug {
			log.Println(errorMsg)
		}
		return Response{
			Messages: []llm.Message{
				{
					Role:    llm.RoleFunction,
					Content: errorMsg,
					Name:    toolName,
				},
			},
		}, nil
	}

	if debug {
		log.Printf("Processing tool call: %s with arguments %v\n", toolName, args)
	}

	// Find the corresponding function in the agent's functions
	var functionFound *AgentFunction
	for _, af := range agent.Functions {
		if af.Name == toolName {
			functionFound = &af
			break
		}
	}

	// Handle case where function is not found
	if functionFound == nil {
		errorMsg := fmt.Sprintf("Error: Tool %s not found", toolName)
		if debug {
			log.Println(errorMsg)
		}
		return Response{
			Messages: []llm.Message{
				{
					Role:    llm.RoleFunction,
					Content: errorMsg,
					Name:    toolName,
				},
			},
		}, nil
	}

	// Execute the function
	result := functionFound.Function(args, contextVariables)

	// Create a message with the tool result
	var resultContent string
	if result.Error != nil {
		resultContent = fmt.Sprintf("Error: %v", result.Error)
	} else {
		resultContent = fmt.Sprintf("%v", result.Data)
	}

	// Create function response message properly formatted for tool call
	toolResultMessage := llm.Message{
		Role:    llm.RoleFunction,
		Content: resultContent,
		Name:    toolName,
	}

	// Return the response with the tool result
	return Response{
		Messages:         []llm.Message{toolResultMessage},
		Agent:            result.Agent,
		ContextVariables: contextVariables,
	}, nil
}

// handleToolCalls handles multiple tool calls with correct context forwarding
func (s *Swarm) handleToolCalls(
	ctx context.Context,
	toolCalls []llm.ToolCall,
	history []llm.Message,
	agent *Agent,
	contextVariables map[string]interface{},
	modelOverride string,
	stream bool,
	debug bool,
	parallel bool,
) ([]ToolResult, []llm.Message, *Agent, error) {
	var toolResults []ToolResult
	updatedAgent := agent
	updatedHistory := make([]llm.Message, len(history))
	copy(updatedHistory, history)

	// Execute tools sequentially for now for simplicity
	for _, toolCall := range toolCalls {
		// Execute the tool call
		toolResp, err := s.handleToolCall(ctx, &toolCall, updatedAgent, contextVariables, debug)
		if err != nil {
			if debug {
				log.Printf("Error executing tool %s: %v", toolCall.Function.Name, err)
			}
			continue
		}

		// Parse arguments for the result
		var args interface{}
		_ = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)

		// Record the tool result
		toolResults = append(toolResults, ToolResult{
			ToolName: toolCall.Function.Name,
			Args:     args,
			Result: Result{
				Success: true,
				Data:    toolResp.Messages[0].Content,
				Error:   nil,
				Agent:   toolResp.Agent,
			},
		})

		// Add the function result to history with proper role and name
		updatedHistory = append(updatedHistory, llm.Message{
			Role:    llm.RoleFunction,
			Content: toolResp.Messages[0].Content,
			Name:    toolCall.Function.Name,
		})

		// Update agent if needed
		if toolResp.Agent != nil {
			updatedAgent = toolResp.Agent
		}
	}

	// Only try to get follow-up if we executed at least one tool successfully
	if len(toolResults) > 0 {
		followUpHistory := withAgentInstructions(
			updatedHistory,
			resolveInstructions(updatedAgent, contextVariables),
			updatedAgent != agent,
		)

		// CRITICAL FIX: Create a new request with the updated history INCLUDING function results
		followUpReq := llm.ChatCompletionRequest{
			Model:    s.resolveModel(updatedAgent, modelOverride),
			Messages: followUpHistory,
			Tools:    buildToolsForAgent(updatedAgent),
		}

		followUpClient, err := s.resolveClient(updatedAgent)
		if err != nil {
			return nil, updatedHistory, updatedAgent, err
		}

		// Create a timeout for the follow-up request
		followUpCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		if debug {
			log.Printf("Getting follow-up with %d messages in history", len(followUpHistory))

			// Print last few messages for debugging
			lastN := 3
			if len(followUpHistory) < lastN {
				lastN = len(followUpHistory)
			}

			log.Printf("Last %d messages:", lastN)
			for i := len(followUpHistory) - lastN; i < len(followUpHistory); i++ {
				msg := followUpHistory[i]
				log.Printf("[%s] %s: %s", msg.Role, msg.Name, truncateString(msg.Content, 50))
			}
		}

		// Get the follow-up response with proper context
		followUpResp, err := followUpClient.CreateChatCompletion(followUpCtx, followUpReq)

		if err != nil {
			if debug {
				log.Printf("Error getting follow-up: %v", err)
			}
			// Continue without follow-up rather than failing
		} else if len(followUpResp.Choices) > 0 {
			// Add follow-up response to history
			followUpMessage := followUpResp.Choices[0].Message

			// Only add if it has content and only append content (no tools)
			if followUpMessage.Content != "" {
				cleanedFollowUp := llm.Message{
					Role:    followUpMessage.Role,
					Content: followUpMessage.Content,
				}
				updatedHistory = append(updatedHistory, cleanedFollowUp)

				if debug {
					log.Printf("Added follow-up: %s", truncateString(followUpMessage.Content, 50))
				}
			} else if debug {
				log.Println("Follow-up was empty")
			}
		}
	}

	return toolResults, updatedHistory, updatedAgent, nil
}

// Helper function to truncate strings for debugging
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// Run is the main entry point for agent execution
func (s *Swarm) Run(
	ctx context.Context,
	agent *Agent,
	messages []llm.Message,
	contextVariables map[string]interface{},
	modelOverride string,
	stream bool,
	debug bool,
	maxTurns int,
	executeTools bool,
) (Response, error) {
	// Validate inputs
	if agent == nil {
		return Response{}, fmt.Errorf("agent cannot be nil")
	}

	// Use a cloned copy of messages for history
	history := make([]llm.Message, len(messages))
	copy(history, messages)

	if contextVariables == nil {
		contextVariables = make(map[string]interface{})
	}

	history = withAgentInstructions(history, resolveInstructions(agent, contextVariables), false)

	// Get chat completion from LLM
	if debug {
		log.Printf("Getting initial response with %d messages", len(history))
	}

	model := s.resolveModel(agent, modelOverride)

	// Create the request
	req := llm.ChatCompletionRequest{
		Model:    model,
		Messages: history,
		Tools:    buildToolsForAgent(agent),
	}

	client, err := s.resolveClient(agent)
	if err != nil {
		return Response{}, err
	}

	// Get initial response
	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return Response{}, fmt.Errorf("chat completion error: %v", err)
	}

	if len(resp.Choices) == 0 {
		return Response{}, fmt.Errorf("no choices in response")
	}

	// Extract the response
	choice := resp.Choices[0]
	history = append(history, choice.Message)

	// Handle tool calls if present and execution is enabled
	if len(choice.Message.ToolCalls) > 0 && executeTools {
		if debug {
			log.Printf("Handling %d tool calls", len(choice.Message.ToolCalls))
		}

		// Execute tools and get the updated history including follow-up
		toolResults, updatedHistory, updatedAgent, err := s.handleToolCalls(
			ctx, choice.Message.ToolCalls, history, agent,
			contextVariables, modelOverride, stream, debug,
			agent.ParallelToolCalls)

		if err != nil {
			return Response{}, fmt.Errorf("tool execution error: %v", err)
		}

		// Calculate which messages to return (only the new ones)
		newMessages := updatedHistory[len(messages):]

		return Response{
			Messages:         newMessages,
			Agent:            updatedAgent,
			ContextVariables: contextVariables,
			ToolResults:      toolResults,
		}, nil
	}

	// No tool calls - just return the normal response
	return Response{
		Messages:         history[len(messages):],
		Agent:            agent,
		ContextVariables: contextVariables,
	}, nil
}

// handleToolCallsParallel executes multiple tool calls concurrently
func (s *Swarm) handleToolCallsParallel(
	ctx context.Context,
	toolCalls []llm.ToolCall,
	history []llm.Message,
	agent *Agent,
	contextVariables map[string]interface{},
	modelOverride string,
	stream bool,
	debug bool,
) ([]ToolResult, []llm.Message, *Agent, error) {
	type toolCallResult struct {
		index  int
		result Response
		err    error
	}

	resultChan := make(chan toolCallResult, len(toolCalls))
	updatedHistory := make([]llm.Message, len(history))
	copy(updatedHistory, history)
	updatedAgent := agent

	// Create a cancellable context for all tool calls
	execCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Launch goroutines for each tool call
	for i, toolCall := range toolCalls {
		go func(idx int, tc llm.ToolCall) {
			toolResp, err := s.handleToolCall(execCtx, &tc, agent, contextVariables, debug)
			resultChan <- toolCallResult{index: idx, result: toolResp, err: err}
		}(i, toolCall)
	}

	// Collect results
	var toolResults []ToolResult
	agentTransferred := false

	for i := 0; i < len(toolCalls); i++ {
		select {
		case <-ctx.Done():
			return nil, history, agent, ctx.Err()
		case result := <-resultChan:
			if result.err != nil {
				if debug {
					log.Printf("Error in tool call %d: %v", result.index, result.err)
				}
				continue
			}

			// Get the original tool call
			toolCall := toolCalls[result.index]
			var args interface{}
			_ = json.Unmarshal([]byte(toolCall.Function.Arguments), &args)

			// Add to tool results
			toolResults = append(toolResults, ToolResult{
				ToolName: toolCall.Function.Name,
				Args:     args,
				Result: Result{
					Success: true,
					Data:    result.result.Messages[0].Content,
					Error:   nil,
					Agent:   result.result.Agent,
				},
			})

			// Add to history
			updatedHistory = append(updatedHistory, llm.Message{
				Role:    llm.RoleFunction,
				Content: result.result.Messages[0].Content,
				Name:    toolCall.Function.Name,
			})

			// Only update agent if not already transferred
			if result.result.Agent != nil && !agentTransferred {
				updatedAgent = result.result.Agent
				agentTransferred = true
			}

			// Store in memory
			if agent.Memory != nil {
				agent.Memory.AddMemory(Memory{
					Content: fmt.Sprintf("Tool %s call with args: %v, result: %s",
						toolCall.Function.Name, args, result.result.Messages[0].Content),
					Type:       "tool_call",
					Context:    map[string]interface{}{"tool": toolCall.Function.Name},
					Timestamp:  time.Now(),
					Importance: 0.7,
				})
			}
		}
	}

	return toolResults, updatedHistory, updatedAgent, nil
}
