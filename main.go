package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/budgies-nest/budgie/agents"
	"github.com/openai/openai-go"
)

func main() {

	userMessage := `
	Add 10 and 32
	Add 12 and 30
	Say Hello to Bob
	Add 40 and 2
	Add 5 and 37	
	Multiply 5 and 6
	Say Hello to Alice
	Multiply 10 and 3				
	`

	// ai/gemma3n

	configs := []Config{
		{
			Info:    "DMR Qwen2.5 [7.62B IQ2_XXS/Q4_K_M]",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "ai/qwen2.5:latest",
		},
		{
			Info:    "Ollama Qwen2.5 [7b]",
			BaseURL: "http://localhost:11434/v1",
			Model:   "qwen2.5:latest",
		},
		{
			Info:    "DMR Qwen3 [8.19B IQ2_XXS/Q4_K_M]",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "ai/qwen3:latest",
		},
		{
			Info:    "Ollama Qwen3 [8b]",
			BaseURL: "http://localhost:11434/v1",
			Model:   "qwen3:8b",
		},
		{
			Info:    "DMR Qwen3 [751.63M IQ2_XXS/Q4_K_M]",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "ai/qwen3:0.6B-Q4_K_M",
		},
		{
			Info:    "Ollama Qwen3 [0.6b]",
			BaseURL: "http://localhost:11434/v1",
			Model:   "qwen3:0.6b",
		},
		{
			Info:    "DMR Gemma3n [6.87B IQ2_XXS/Q4_K_M]",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "ai/gemma3n:latest",
		},
		{
			Info:    "DMR Llama-xLAM-2 q4_k_m",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "hf.co/salesforce/llama-xlam-2-8b-fc-r-gguf:q4_k_m",
		},
		{
			Info:    "DMR Llama-xLAM-2 q2_k",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "hf.co/salesforce/llama-xlam-2-8b-fc-r-gguf:q2_k",
		},
		{
			Info:    "DMR xlam-2-3b-fc-r-gguf:q4_k_m",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "hf.co/salesforce/xlam-2-3b-fc-r-gguf:q4_k_m",
		},
		{
			Info:    "DMR xlam-2-3b-fc-r-gguf:q4_k_s",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "hf.co/salesforce/xlam-2-3b-fc-r-gguf:q4_k_s",
		},
		{
			Info:    "DMR xlam-2-3b-fc-r-gguf:q4_0",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "hf.co/salesforce/xlam-2-3b-fc-r-gguf:q4_0",
		},
		{
			Info:    "DMR xlam-2-3b-fc-r-gguf:q3_k_l",
			BaseURL: "http://localhost:12434/engines/llama.cpp/v1",
			Model:   "hf.co/salesforce/xlam-2-3b-fc-r-gguf:q3_k_l",
		},

	} 

	gpuInfo := GetGPUInfo()
	fmt.Println(gpuInfo)
	results := RunToolsCompletions(userMessage, configs)
	report := GenerateTabularReport(results, gpuInfo)
	fmt.Print(report)
	SaveReportToFile(report, gpuInfo)

}

type Config struct {
	Info    string
	BaseURL string
	Model   string
}

type CompletionResult struct {
	Config            Config
	DetectedToolCalls []openai.ChatCompletionMessageToolCall
	Results           []interface{}
	ToolCallDetails   []ToolCallDetail
	Duration          time.Duration
	Error             error
}

type ToolCallDetail struct {
	Name      string
	Arguments string
	Result    interface{}
}

func RunToolsCompletions(userMessage string, configurations []Config) []CompletionResult {

	fmt.Println("Running tool completions:")
	var results []CompletionResult

	for _, config := range configurations {

		fmt.Printf("Running for: %s\n", config.Info)
		start := time.Now()

		result := CompletionResult{
			Config: config,
		}

		agent, err := GetAgent(config.BaseURL, config.Model, userMessage)
		if err != nil {
			fmt.Println("Error getting agent:", err)
			result.Error = err
			result.Duration = time.Since(start)
			results = append(results, result)
			continue
		}

		detectedToolCalls, err := ToolsCompletion(agent)
		if err != nil {
			fmt.Println("Error generating tool calls:", err)
			result.Error = err
			result.Duration = time.Since(start)
			results = append(results, result)
			continue
		}

		result.DetectedToolCalls = detectedToolCalls
		fmt.Printf("Number of detected tool calls for %s: %d\n", config.Model, len(detectedToolCalls))

		executionResults, err := agent.ExecuteToolCalls(detectedToolCalls, ToolCallHandlers())
		if err != nil {
			fmt.Println("Error executing tool calls:", err)
			result.Error = err
		} else {
			result.Results = make([]interface{}, len(executionResults))
			result.ToolCallDetails = make([]ToolCallDetail, len(detectedToolCalls))

			for i, r := range executionResults {
				result.Results[i] = r
			}

			// Create tool call details with arguments and results
			for i, toolCall := range detectedToolCalls {
				result.ToolCallDetails[i] = ToolCallDetail{
					Name:      toolCall.Function.Name,
					Arguments: toolCall.Function.Arguments,
					Result:    nil, // Will be set below
				}
				if i < len(executionResults) {
					result.ToolCallDetails[i].Result = executionResults[i]
				}
			}
		}

		result.Duration = time.Since(start)
		results = append(results, result)
	}

	return results
}

func GetAgent(baseUrl string, model string, userMessage string) (*agents.Agent, error) {
	toolCallsAgent, err := agents.NewAgent("tool_calls_agent",

		agents.WithOpenAIClient("my_api_key", baseUrl),

		agents.WithParams(
			openai.ChatCompletionNewParams{
				Model:       model,
				Temperature: openai.Opt(0.0),
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage(userMessage),
				},
				ParallelToolCalls: openai.Bool(true),
			},
		),
		agents.WithTools(ToolsCatalog()),
	)
	if err != nil {
		return nil, err
	}
	return toolCallsAgent, nil
}

func ToolsCompletion(agent *agents.Agent) ([]openai.ChatCompletionMessageToolCall, error) {

	// Generate the tools detection completion
	detectedToolCalls, err := agent.ToolsCompletion(context.Background())

	if err != nil {
		return nil, fmt.Errorf("error generating tool calls: %w", err)
	}

	return detectedToolCalls, nil
}

func ToolsCatalog() []openai.ChatCompletionToolParam {
	addTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "add",
			Description: openai.String("add two numbers"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"a": map[string]string{
						"type":        "number",
						"description": "The first number to add.",
					},
					"b": map[string]string{
						"type":        "number",
						"description": "The second number to add.",
					},
				},
				"required": []string{"a", "b"},
			},
		},
	}

	multiplicationTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "multiply",
			Description: openai.String("multiply two numbers"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"a": map[string]string{
						"type":        "number",
						"description": "The first number to multiply.",
					},
					"b": map[string]string{
						"type":        "number",
						"description": "The second number to multiply.",
					},
				},
				"required": []string{"a", "b"},
			},
		},
	}

	sayHelloTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "say_hello",
			Description: openai.String("Say hello to the given person name"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]string{
						"type": "string",
					},
				},
				"required": []string{"name"},
			},
		},
	}

	return []openai.ChatCompletionToolParam{addTool, sayHelloTool, multiplicationTool}
}

func ToolCallHandlers() map[string]func(any) (any, error) {

	return map[string]func(any) (any, error){
		"add": func(args any) (any, error) {
			a := args.(map[string]any)["a"].(float64)
			b := args.(map[string]any)["b"].(float64)
			result := a + b
			fmt.Println(" - add tool with args:", args, "result:", result)
			return result, nil

		},

		"say_hello": func(args any) (any, error) {
			name := args.(map[string]any)["name"].(string)
			result := fmt.Sprintf("Hello, %s!", name)
			fmt.Println(" - say_hello tool with args:", args, "result:", result)
			return result, nil
		},

		"multiply": func(args any) (any, error) {
			a := args.(map[string]any)["a"].(float64)
			b := args.(map[string]any)["b"].(float64)
			result := a * b
			fmt.Println(" - multiply tool with args:", args, "result:", result)
			return result, nil
		},
	}
}

type GPUInfo struct {
	ChipsetModel string
	VRAM         string
	Vendor       string
}

func GetGPUInfo() GPUInfo {
	gpuInfo := GPUInfo{}
	cmd := exec.Command("system_profiler", "SPDisplaysDataType")
	output, err := cmd.Output()
	if err == nil {
		lines := strings.SplitSeq(string(output), "\n")
		for line := range lines {
			line = strings.TrimSpace(line)
			if strings.Contains(line, "Chipset Model:") {
				gpuInfo.ChipsetModel = strings.TrimPrefix(line, "Chipset Model: ")
			}
			if strings.Contains(line, "VRAM (Total):") {
				gpuInfo.VRAM = strings.TrimPrefix(line, "VRAM (Total): ")
			}
			if strings.Contains(line, "Vendor:") {
				gpuInfo.Vendor = strings.TrimPrefix(line, "Vendor: ")
			}
		}
	}
	if gpuInfo.ChipsetModel == "" {
		gpuInfo.ChipsetModel = "Unknown"
	}
	if gpuInfo.VRAM == "" {
		gpuInfo.VRAM = "Unknown"
	}
	if gpuInfo.Vendor == "" {
		gpuInfo.Vendor = "Unknown"
	}
	return gpuInfo
}

func GenerateTabularReport(results []CompletionResult, gpuInfo GPUInfo) string {
	var report strings.Builder

	report.WriteString("\n=== BENCHMARK REPORT ===\n")
	report.WriteString(fmt.Sprintf("GPU: %s (%s) - %s\n", gpuInfo.ChipsetModel, gpuInfo.Vendor, gpuInfo.VRAM))
	report.WriteString(fmt.Sprintf("%-35s | %-15s | %-20s | %-15s | %-10s | %-25s | %-15s\n", "", "", "", "", "", "", ""))

	// Header
	report.WriteString(fmt.Sprintf("%-35s | %-15s | %-20s | %-15s | %-10s | %-25s | %-15s\n",
		"Configuration", "Tool Calls", "Results", "Duration", "Status", "Arguments", "Result"))
	report.WriteString(strings.Repeat("-", 140) + "\n")

	// Results
	for _, result := range results {
		status := "SUCCESS"
		if result.Error != nil {
			status = "ERROR"
		}

		toolCallsStr := fmt.Sprintf("%d calls", len(result.DetectedToolCalls))
		resultsStr := fmt.Sprintf("%d results", len(result.Results))
		durationStr := fmt.Sprintf("%.2fs", result.Duration.Seconds())

		report.WriteString(fmt.Sprintf("%-35s | %-15s | %-20s | %-15s | %-10s | %-25s | %-15s\n",
			result.Config.Info, toolCallsStr, resultsStr, durationStr, status, "", ""))

		// Show tool call details with arguments and results
		for i, detail := range result.ToolCallDetails {
			// Format arguments (remove outer braces and quotes for cleaner display)
			args := strings.ReplaceAll(detail.Arguments, "\"", "")
			if len(args) > 25 {
				args = args[:22] + "..."
			}

			// Format result
			resultStr := fmt.Sprintf("%v", detail.Result)
			if len(resultStr) > 15 {
				resultStr = resultStr[:12] + "..."
			}

			report.WriteString(fmt.Sprintf("  %-33s | %-15s | %-20s | %-15s | %-10s | %-25s | %-15s\n",
				fmt.Sprintf("%d. %s", i+1, detail.Name), "", "", "", "", args, resultStr))
		}

		// Show error if any
		if result.Error != nil {
			report.WriteString(fmt.Sprintf("     %-31s | %-15s | %-20s | %-15s | %-10s | %-25s | %-15s\n",
				fmt.Sprintf("Error: %s", result.Error.Error()), "", "", "", "", "", ""))
		}

		report.WriteString(fmt.Sprintf("%-35s | %-15s | %-20s | %-15s | %-10s | %-25s | %-15s\n", "", "", "", "", "", "", ""))
	}

	return report.String()
}

func SaveReportToFile(report string, gpuInfo GPUInfo) {
	// Create filename with GPU info and current date/time
	now := time.Now()

	// Clean GPU model name for filename (remove spaces and special characters)
	cleanGPUModel := strings.ReplaceAll(gpuInfo.ChipsetModel, " ", "_")
	cleanGPUModel = strings.ReplaceAll(cleanGPUModel, "(", "")
	cleanGPUModel = strings.ReplaceAll(cleanGPUModel, ")", "")
	cleanGPUModel = strings.ReplaceAll(cleanGPUModel, "/", "_")

	// Format: GPU_Model_YYYY-MM-DD_HH-MM-SS.txt
	filename := fmt.Sprintf("%s_%s.txt",
		cleanGPUModel,
		now.Format("2006-01-02_15-04-05"))

	// Write report to file
	err := os.WriteFile(filename, []byte(report), 0644)
	if err != nil {
		fmt.Printf("Error saving report to file: %v\n", err)
		return
	}

	fmt.Printf("Report saved to: %s\n", filename)
}
