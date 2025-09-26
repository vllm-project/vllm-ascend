package metrics_collector

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// 对目标metrics数据执行histogram combine操作
func combineHistogramMetrics(c *Collector, metricsName string, role string) error {
	sampleCount := uint64(0)
	sampleSum := 0.0
	buckets := make(map[float64]uint64)
	var labels = prometheus.Labels{}

	for _, instance := range c.instances {
		// 实例身份需要为目标身份才会被统计， 注： api_server包括所有的P、D实例
		if strings.ToLower(instance.Role) == role || (role == "api_server" && strings.ToLower(instance.Role) != "scheduler") {
			metrics := c.histogramMetrics[metricsName]
			allLabels := instance.getInstanceLabelsForMetrics(metricsName)
			for _, label := range allLabels {
				if len(labels) == 0 {
					labels = filterCustomLabels(label)
					labels["engine"] = "All"
				}
				histogramData := metrics.GetHistogramData(label)
				sampleCount += histogramData.SampleCount
				sampleSum += histogramData.SampleSum
				for upperBound, count := range histogramData.Buckets {
					buckets[upperBound] += count
				}
			}
		}
	}

	if len(labels) > 0 {
		// 获取到目标数据则写入普罗
		c.aggregatedHistogramMetrics[metricsName].SetHistogramData(labels, &HistogramData{
			SampleCount: sampleCount,
			SampleSum:   sampleSum,
			Buckets:     buckets,
		})
	}
	return nil
}

// 对目标metrics执行sum操作
func sumMetrics(c *Collector, metricsName string, metricsType string, role string) error {
	if metricsType == "Gauge" {
		err := sumGaugeMetrics(c, metricsName, role)
		if err != nil {
			return err
		}
	} else if metricsType == "Counter" {
		err := sumCounterMetrics(c, metricsName, role)
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("metrics type %s is not supported", metricsType)
	}
	return nil
}

// 执行Gauge类型的metrics数据的sum操作
func sumGaugeMetrics(c *Collector, metricsName string, role string) error {
	gaugeCount := float64(0)
	var labels = prometheus.Labels{}
	metrics := c.gaugeMetrics[metricsName]

	for _, instance := range c.instances {
		// 实例身份需要为目标身份才会被统计， 注： api_server包括所有的P、D实例
		if strings.ToLower(instance.Role) == role || (role == "api_server" && strings.ToLower(instance.Role) != "scheduler") {
			allLabels := instance.getInstanceLabelsForMetrics(metricsName)
			for _, label := range allLabels {
				if len(labels) == 0 {
					labels = filterCustomLabels(label)
					labels["engine"] = "All"
				}
				value := getMetricsValueFromSpecificLabels(label, metrics)
				gaugeCount += value
			}
		}
	}

	if len(labels) > 0 {
		// 获取到目标数据则写入普罗
		c.aggregatedGaugeMetrics[metricsName].With(labels).Set(gaugeCount)
	}
	return nil
}

// 执行Counter类型的metrics数据的sum操作
func sumCounterMetrics(c *Collector, metricsName string, role string) error {
	metrics := c.counterMetrics[metricsName]

	targetLabels := getTargetLabels(c.instances, metricsName)

	for _, targetLabel := range targetLabels {

		var labels = prometheus.Labels{}
		counterCount := float64(0)

		for _, instance := range c.instances {
			// 实例身份需要为目标身份才会被统计， 注： api_server包括所有的P、D实例
			if strings.ToLower(instance.Role) == role ||
				(role == "api_server" && strings.ToLower(instance.Role) != "scheduler") ||
				optionForRequestSuccess(role, strings.ToLower(instance.Role), metricsName, targetLabel) {

				allLabels := instance.getInstanceLabelsForMetrics(metricsName)
				for _, label := range allLabels {
					if !labelsMatch(label, targetLabel) {
						continue
					}
					if len(labels) == 0 {
						labels = filterCustomLabels(label)
						labels["engine"] = "All"
					}
					value := getMetricsValueFromSpecificLabels(label, metrics)
					counterCount += value
				}
			}
		}
		if len(labels) > 0 {
			// 获取到目标数据则写入普罗
			setAggCounterValue(c, metricsName, labels, counterCount)
		}
	}
	return nil
}

// 针对vllm:request_success_total指标的特殊处理
func optionForRequestSuccess(role string, instanceRole string, metricsName string, targetLabel prometheus.Labels) bool {
	if role != "option" {
		return false
	}
	if metricsName != "vllm:request_success_total" {
		return false
	}

	//  finished_reason="abort" -> 统计P+D
	if labelsMatch(targetLabel, prometheus.Labels{"finished_reason": "abort"}) && instanceRole != "scheduler" {
		return true
	}

	// finished_reason="length" -> 统计P+D
	if labelsMatch(targetLabel, prometheus.Labels{"finished_reason": "length"}) && instanceRole != "scheduler" {
		return true
	}

	// finished_reason="stop" -> 统计D
	if labelsMatch(targetLabel, prometheus.Labels{"finished_reason": "stop"}) && instanceRole == "decode" {
		return true
	}

	return false
}

// 过滤自定义label, 还原为初始label
// 注： 默认认为instance、role为自定义的后加入label, 需要剔除
func filterCustomLabels(src prometheus.Labels) prometheus.Labels {
	dst := prometheus.Labels{}
	for k, v := range src {
		if k == "instance" || k == "role" {
			continue
		}
		dst[k] = v
	}
	return dst
}

func isEmptyInstance(instance Instance) bool {
	return reflect.DeepEqual(instance, Instance{})
}

// 获取需要进行筛选的目标指标
// 注： 默认认为instance、role、engine为需要过滤掉的变量label，剩下的label则为目标label
func getTargetLabels(instances []Instance, metricsName string) []prometheus.Labels {
	var apiServerInstance Instance
	labelsList := make([]prometheus.Labels, 0)

	for _, instance := range instances {
		if strings.ToLower(instance.Role) == "prefill" || strings.ToLower(instance.Role) == "decode" {
			apiServerInstance = instance
			break
		}
	}

	if isEmptyInstance(apiServerInstance) {
		return labelsList
	}

	allLabels := apiServerInstance.getInstanceLabelsForMetrics(metricsName)
	for _, label := range allLabels {
		targetLabel := filterTargetLabels(label)
		labelsList = append(labelsList, targetLabel)
	}
	return labelsList
}

func filterTargetLabels(src prometheus.Labels) prometheus.Labels {
	dst := prometheus.Labels{}

	for k, v := range src {
		if k == "instance" || k == "role" || k == "engine" {
			continue
		}
		dst[k] = v
	}
	return dst
}

// 判断目标label是否包含在给定的label中
func labelsMatch(fullLabels, targetLabels prometheus.Labels) bool {
	for key, value := range targetLabels {
		if fullLabelValue, exists := fullLabels[key]; !exists || fullLabelValue != value {
			return false
		}
	}
	return true
}

// 通过labels获取对应的metrics值
func getMetricsValueFromSpecificLabels(labels prometheus.Labels, sourceMetricVec interface{}) float64 {
	switch metricVec := sourceMetricVec.(type) {
	case *prometheus.GaugeVec:
		gaugeMetrics := metricVec.With(labels)
		metric := &dto.Metric{}
		err := gaugeMetrics.Write(metric)
		if err == nil && metric.Gauge != nil {
			value := metric.Gauge.GetValue()
			return value
		}
	case *prometheus.CounterVec:
		counterMetrics := metricVec.With(labels)
		metric := &dto.Metric{}
		err := counterMetrics.Write(metric)
		if err == nil && metric.Counter != nil {
			value := metric.Counter.GetValue()
			return value
		}
	default:
		return 0.0
	}
	return 0.0
}

// 记录counter类型的metrics数据
func recordCounterMetrics(c *Collector, name string, labels prometheus.Labels, metric *dto.Metric) {
	if _, ok := c.counterMetrics[name]; ok && metric.Counter != nil {
		// 原生Counter类型只支持Add方法达到单调递增
		// 通过计算增长差值实现
		currentValue := metric.Counter.GetValue()
		setCounterValue(c, name, labels, currentValue)
	}
}

// 设置Counter类型数据的值
func setCounterValue(c *Collector, name string, labels prometheus.Labels, value float64) {
	writeToCounterMetrics(c.counterMetrics, name, labels, value)
}

// 设置融合之后的Counter类型数据的值
func setAggCounterValue(c *Collector, name string, labels prometheus.Labels, value float64) {
	writeToCounterMetrics(c.aggregatedCounterMetrics, name, labels, value)
}

// Counter指标写入普罗
func writeToCounterMetrics(counterMetrics map[string]*prometheus.CounterVec, name string, labels prometheus.Labels,
	value float64) {
	if counter, ok := counterMetrics[name]; ok && value >= 0 {
		counterInstance := counter.With(labels)
		counterMetrics := &dto.Metric{}
		err := counterInstance.Write(counterMetrics)
		if err == nil && counterMetrics.Counter != nil {
			storedValue := counterMetrics.Counter.GetValue()
			delta := value - storedValue
			if delta >= 0 {
				counterInstance.Add(delta)
			}
		} else {
			counterInstance.Add(value)
		}
	}
}
