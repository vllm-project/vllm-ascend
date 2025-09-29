package metrics_collector

import (
	"os"
	"testing"
)

func TestLoadMetricsYamlConfig(t *testing.T) {
	// 创建临时文件
	tmpfile, err := os.CreateTemp("", "testfile")
	if err != nil {
		t.Fatal(err)
	}
	defer tmpfile.Close()           // 关闭临时文件
	defer os.Remove(tmpfile.Name()) // 删除临时文件

	// 写入测试数据
	_, err = tmpfile.Write([]byte(`configurations:
  - metric_name: metric1
    acting_instance: prefill
    operation: sum
    type: Gauge
  - metric_name: metric2
    acting_instance: decode
    operation: union
    type: Counter`))
	if err != nil {
		t.Fatal(err)
	}

	// 测试文件不存在的情况
	_, err = loadMetricsYamlConfig("nonexistentfile")
	if err == nil {
		t.Error("Expected an error when file does not exist")
	}

	// 测试正常情况
	config, err := loadMetricsYamlConfig(tmpfile.Name())
	if err != nil {
		t.Error(err)
	}
	if len(config.Configurations) != 2 {
		t.Errorf("Expected 2 configurations, got %d", len(config.Configurations))
	}
	if config.Configurations[0].MetricsName != "metric1" {
		t.Errorf("Expected metric1, got %s", config.Configurations[0].MetricsName)
	}
	if config.Configurations[1].Operation != "union" {
		t.Errorf("Expected union, got %s", config.Configurations[1].Operation)
	}
}
