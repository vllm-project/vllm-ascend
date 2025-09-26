package metrics_collector

import (
	"fmt"
	"github.com/go-playground/validator/v10"
	"os"

	"github.com/goccy/go-yaml"
)

// IntegratedConfiguration 定义集成配置结构
type IntegratedConfiguration struct {
	// metrics名称: 如vllm:num_requests_running
	MetricsName string `yaml:"metric_name" validate:"required"`
	// 作用于的实例: prefill, decode, scheduler, api_server等
	ActingInstance string `yaml:"acting_instance" validate:"required,oneof=prefill decode scheduler api_server option"`
	// 操作类型: sum, union, histogram_combine
	Operation string `yaml:"operation" validate:"required,oneof=sum union histogram_combine"`
	// metrics类型: gauge counter histogram
	Type string `yaml:"type" validate:"required,oneof=Gauge Counter Histogram"`
}

// MetricsConfig 顶层配置结构
type MetricsConfig struct {
	MetricsCollectInterval int                       `yaml:"metrics_collect_interval" default:"5"`
	MetricsRequestTimeout  int                       `yaml:"metrics_request_timeout" default:"10"`
	Configurations         []IntegratedConfiguration `yaml:"configurations"`
	metricOperation        map[string]string
}

func loadMetricsYamlConfig(yamlPath string) (*MetricsConfig, error) {
	// 检查文件是否存在
	if _, err := os.Stat(yamlPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file %s does not exist", yamlPath)
	}

	data, err := os.ReadFile(yamlPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %v", yamlPath, err)
	}

	var config MetricsConfig
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse YAML in file %s: %v", yamlPath, err)
	}

	// 初始化验证器
	validate := validator.New()
	// 自定义校验:检查配置是否合法 & metric_name是否重复
	metricNames := make(map[string]bool)
	for i, configuration := range config.Configurations {
		if err := validate.Struct(configuration); err != nil {
			// 验证失败，输出详细错误信息
			return nil, fmt.Errorf("metrics yaml 配置错误，具体原因是第%d个指标: %v", i+1, err)
		}
		if metricNames[configuration.MetricsName] {
			// 发现重复的metric_name
			return nil, fmt.Errorf("metrics yaml 配置非法："+
				"第%d个metric的metric_name=%s已重复出现", i+1, configuration.MetricsName)
		}
		metricNames[configuration.MetricsName] = true
	}

	metricOperation := make(map[string]string)
	for _, configuration := range config.Configurations {
		metricOperation[configuration.MetricsName] = configuration.Operation
	}
	config.metricOperation = metricOperation

	if config.MetricsRequestTimeout == 0 {
		config.MetricsRequestTimeout = 10
	}

	if config.MetricsCollectInterval == 0 {
		config.MetricsCollectInterval = 5
	}

	return &config, nil
}

// 通过config配置文件定义的操作类型对指定metrics进行操作
func processMetricsAccordingConfiguration(cfg IntegratedConfiguration, c *Collector) error {
	// 根据操作类型执行相应逻辑
	switch cfg.Operation {
	case "sum":
		return performSumOperation(cfg, c)
	case "union":
		// union已在注册场景完成，无需操作
		return nil
	case "histogram_combine":
		return performHistogramCombineOperation(cfg, c)
	default:
		return fmt.Errorf("unknown operation: %s", cfg.Operation)
	}
}

// sum操作
func performSumOperation(cfg IntegratedConfiguration, c *Collector) error {
	// 执行求和逻辑
	err := sumMetrics(c, cfg.MetricsName, cfg.Type, cfg.ActingInstance)
	if err != nil {
		return err
	}
	return nil
}

// histogram combine操作
func performHistogramCombineOperation(cfg IntegratedConfiguration, c *Collector) error {
	// 执行直方图合并逻辑
	err := combineHistogramMetrics(c, cfg.MetricsName, cfg.ActingInstance)
	if err != nil {
		return err
	}

	return nil
}