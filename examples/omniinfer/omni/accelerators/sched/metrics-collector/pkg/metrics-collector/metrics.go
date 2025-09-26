package metrics_collector

import (
	"strings"
	"sync"
)

var (
	staticMap map[string]string
	initOnce  sync.Once
)

// 需要从RouteServer中获取的指标名称和vLLM原生的指标名称的对照
func initStaticMap() {
	staticMap = map[string]string{
		"rs:time_to_first_token_sec":  "vllm:time_to_first_token_seconds",
		"rs:time_per_output_token_ms": "vllm:time_per_output_token_seconds",
		"rs:request_total_time_secs":  "vllm:e2e_request_latency_seconds",
	}
}

func getStaticMap() map[string]string {
	initOnce.Do(initStaticMap)
	return staticMap
}

// 收集时需要增加的labels来区分各个不同的实例
var collectLabels = []string{"role", "instance"}

func getCollectLabels() []string {
	return collectLabels
}

func getRealMetricsName(metricsName string) string {
	// 是从route server获取到的metrics
	if strings.HasPrefix(metricsName, "rs:") {
		name, ok := getStaticMap()[metricsName]
		if !ok {
			return ""
		}
		return name
	}
	return metricsName
}