package metrics_collector

import (
	"reflect"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

func TestOptionForRequestSuccess(t *testing.T) {
	tests := []struct {
		name         string
		role         string
		instanceRole string
		metricsName  string
		targetLabel  map[string]string
		want         bool
	}{
		{
			name:         "role is not option",
			role:         "api_server",
			instanceRole: "scheduler",
			metricsName:  "vllm:request_success_total",
			targetLabel:  map[string]string{"finished_reason": "abort"},
			want:         false,
		},
		{
			name:         "metricsName is not vllm:request_success_total",
			role:         "option",
			instanceRole: "scheduler",
			metricsName:  "not_vllm:request_success_total",
			targetLabel:  map[string]string{"finished_reason": "abort"},
			want:         false,
		},
		{
			name:         "instanceRole is scheduler and finished_reason is abort",
			role:         "option",
			instanceRole: "scheduler",
			metricsName:  "vllm:request_success_total",
			targetLabel:  map[string]string{"finished_reason": "abort"},
			want:         false,
		},
		{
			name:         "instanceRole is not scheduler and finished_reason is abort",
			role:         "option",
			instanceRole: "prefill",
			metricsName:  "vllm:request_success_total",
			targetLabel:  map[string]string{"finished_reason": "abort"},
			want:         true,
		},
		{
			name:         "instanceRole is decode and finished_reason is stop",
			role:         "option",
			instanceRole: "decode",
			metricsName:  "vllm:request_success_total",
			targetLabel:  map[string]string{"finished_reason": "stop"},
			want:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := optionForRequestSuccess(tt.role, tt.instanceRole, tt.metricsName, tt.targetLabel)
			if got != tt.want {
				t.Errorf("optionForRequestSuccess() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFilterCustomLabels(t *testing.T) {
	// 测试用例1：当src为空时，返回的dst也应为空
	t.Run("EmptySrc", func(t *testing.T) {
		src := prometheus.Labels{}
		dst := filterCustomLabels(src)
		if len(dst) != 0 {
			t.Errorf("Expected dst to be empty, but got %v", dst)
		}
	})

	// 测试用例2：当src中包含"instance"和"role"时，返回的dst不应包含这两个标签
	t.Run("SrcContainsInstanceAndRole", func(t *testing.T) {
		src := prometheus.Labels{
			"instance": "localhost:9090",
			"role":     "alertmanager",
			"job":      "prometheus",
		}
		dst := filterCustomLabels(src)
		expected := prometheus.Labels{
			"job": "prometheus",
		}
		if !reflect.DeepEqual(dst, expected) {
			t.Errorf("Expected dst to be %v, but got %v", expected, dst)
		}
	})

	// 测试用例3：当src中不包含"instance"和"role"时，返回的dst应与src相同
	t.Run("SrcNotContainsInstanceAndRole", func(t *testing.T) {
		src := prometheus.Labels{
			"job": "prometheus",
			"env": "production",
		}
		dst := filterCustomLabels(src)
		if !reflect.DeepEqual(dst, src) {
			t.Errorf("Expected dst to be %v, but got %v", src, dst)
		}
	})
}

func TestIsEmptyInstance(t *testing.T) {
	// Test case 1: instance is an empty instance
	{
		instance := Instance{}
		if !isEmptyInstance(instance) {
			t.Errorf("Expected true, got false")
		}
	}

	// Test case 2: instance is not an empty instance
	{
		// Assuming Instance has a field named 'Field'
		instance := Instance{Role: "Prefill"}
		if isEmptyInstance(instance) {
			t.Errorf("Expected false, got true")
		}
	}
}

func TestGetTargetLabels(t *testing.T) {
	// 测试用例1：当实例列表为空时，应返回空列表
	var instances []Instance
	metricsName := "testMetrics"
	labels := getTargetLabels(instances, metricsName)
	if len(labels) != 0 {
		t.Errorf("Expected an empty list, but got %v", labels)
	}

	// 测试用例2：当实例列表中没有"prefill"或"decode"角色的实例时，应返回空列表
	instances = []Instance{
		{
			Role: "role1",
		},
		{
			Role: "role2",
		},
	}
	labels = getTargetLabels(instances, metricsName)
	if len(labels) != 0 {
		t.Errorf("Expected an empty list, but got %v", labels)
	}

	// 测试用例3：当实例列表中存在"prefill"或"decode"角色的实例时，应返回非空列表
	instances = []Instance{
		{
			Role: "Decode",
			MetricsLabels: map[string]map[string]prometheus.Labels{
				"testMetrics": {
					"engine=31|finished_reason=abort|instance=127.0.0.1:9116|model_name=deepseek|role=decode": {
						"engine":          "31",
						"instance":        "127.0.0.1:9116",
						"model_name":      "deepseek",
						"decode":          "decode",
						"finished_reason": "abort",
					},
				},
			},
		},
	}
	labels = getTargetLabels(instances, metricsName)
	if len(labels) == 0 {
		t.Errorf("Expected a non-empty list, but got %v", labels)
	}
}

func TestFilterTargetLabels(t *testing.T) {
	// 测试用例1：源标签中没有需要过滤的标签
	src := prometheus.Labels{
		"job": "prometheus",
		"env": "production",
	}
	dst := filterTargetLabels(src)
	if len(dst) != len(src) {
		t.Errorf("Expected %d labels, got %d", len(src), len(dst))
	}

	// 测试用例2：源标签中有一个需要过滤的标签
	src = prometheus.Labels{
		"job":      "prometheus",
		"instance": "localhost:9090",
	}
	dst = filterTargetLabels(src)
	if len(dst) != len(src)-1 {
		t.Errorf("Expected %d labels, got %d", len(src)-1, len(dst))
	}

	// 测试用例3：源标签中有多个需要过滤的标签
	src = prometheus.Labels{
		"job":      "prometheus",
		"instance": "localhost:9090",
		"role":     "server",
		"engine":   "docker",
	}
	dst = filterTargetLabels(src)
	if len(dst) != len(src)-3 {
		t.Errorf("Expected %d labels, got %d", len(src)-3, len(dst))
	}

	// 测试用例4：源标签中所有标签都需要过滤
	src = prometheus.Labels{
		"instance": "localhost:9090",
		"role":     "server",
		"engine":   "docker",
	}
	dst = filterTargetLabels(src)
	if len(dst) != 0 {
		t.Errorf("Expected %d labels, got %d", 0, len(dst))
	}
}

func TestLabelsMatch(t *testing.T) {
	// 测试用例1：当目标标签是空的时候，无论全标签是什么，都应该返回true
	{
		fullLabels := prometheus.Labels{
			"key1": "value1",
			"key2": "value2",
		}
		targetLabels := prometheus.Labels{}
		if !labelsMatch(fullLabels, targetLabels) {
			t.Errorf("Want true, got false")
		}
	}

	// 测试用例2：当目标标签的键在全标签中不存在时，应返回false
	{
		fullLabels := prometheus.Labels{
			"key1": "value1",
			"key2": "value2",
		}
		targetLabels := prometheus.Labels{
			"key3": "value3",
		}
		if labelsMatch(fullLabels, targetLabels) {
			t.Errorf("Want false, got true")
		}
	}

	// 测试用例3：当目标标签的键在全标签中存在，但值不匹配时，应返回false
	{
		fullLabels := prometheus.Labels{
			"key1": "value1",
			"key2": "value2",
		}
		targetLabels := prometheus.Labels{
			"key1": "value3",
		}
		if labelsMatch(fullLabels, targetLabels) {
			t.Errorf("Want false, got true")
		}
	}

	// 测试用例4：当目标标签的键在全标签中存在，且值匹配时，应返回true
	{
		fullLabels := prometheus.Labels{
			"key1": "value1",
			"key2": "value2",
		}
		targetLabels := prometheus.Labels{
			"key1": "value1",
		}
		if !labelsMatch(fullLabels, targetLabels) {
			t.Errorf("Want true, got false")
		}
	}

	// 测试用例5：当目标标签的键在全标签中存在，且值匹配，但全标签有额外的键值对时，应返回true
	{
		fullLabels := prometheus.Labels{
			"key1": "value1",
			"key2": "value2",
			"key3": "value3",
		}
		targetLabels := prometheus.Labels{
			"key1": "value1",
		}
		if !labelsMatch(fullLabels, targetLabels) {
			t.Errorf("Want true, got false")
		}
	}
}
