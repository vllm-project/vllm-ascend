package metrics_collector

import (
	"github.com/prometheus/client_golang/prometheus"
	"strings"
	"sync"
	"time"
)

type CustomHistogramData struct {
	SampleCount uint64
	SampleSum   float64
	Buckets     map[float64]uint64
	LastUpdate  time.Time
}

type CustomHistogram struct {
	mu         sync.Mutex
	data       map[string]*CustomHistogramData
	opts       prometheus.HistogramOpts
	labelNames []string
	// 缓存的 Desc 实例
	desc *prometheus.Desc
}

type HistogramData struct {
	SampleCount uint64
	SampleSum   float64
	Buckets     map[float64]uint64
}

func NewCustomHistogram(opts prometheus.HistogramOpts, labelNames []string) *CustomHistogram {
	return &CustomHistogram{
		data:       make(map[string]*CustomHistogramData),
		opts:       opts,
		labelNames: labelNames,
		desc: prometheus.NewDesc(
			opts.Name,
			opts.Help,
			labelNames,
			opts.ConstLabels,
		),
	}
}

func (h *CustomHistogram) SetHistogramData(labels prometheus.Labels, data *HistogramData) {
	h.mu.Lock()
	defer h.mu.Unlock()

	labelKeys := h.createLabelKey(labels)

	// 尝试直接把值设置进去
	h.data[labelKeys] = &CustomHistogramData{
		SampleCount: data.SampleCount,
		SampleSum:   data.SampleSum,
		Buckets:     make(map[float64]uint64),
		LastUpdate:  time.Now(),
	}

	//处理bucket中的信息
	for k, v := range data.Buckets {
		h.data[labelKeys].Buckets[k] = v
	}
}

func (h *CustomHistogram) GetHistogramData(labels prometheus.Labels) *CustomHistogramData {
	h.mu.Lock()
	defer h.mu.Unlock()

	labelKeys := h.createLabelKey(labels)
	if v, ok := h.data[labelKeys]; ok {
		return v
	}
	return nil
}

func (h *CustomHistogram) GetAllData() map[string]*CustomHistogramData {
	h.mu.Lock()
	defer h.mu.Unlock()

	result := make(map[string]*CustomHistogramData)
	for k, v := range h.data {
		result[k] = v
	}
	return result
}

// 创建标签键
func (h *CustomHistogram) createLabelKey(labels prometheus.Labels) string {
	key := ""
	for _, name := range h.labelNames {
		if value, exists := labels[name]; exists {
			key += name + "=" + value + ","
		}
	}
	return key
}

// 解析标签
func (h *CustomHistogram) parseLabelKey(labelKey string) prometheus.Labels {
	labels := prometheus.Labels{}

	// 标签示例: engine="0",le="0.001",model_name="deepseek"
	parts := strings.Split(labelKey, ",")
	for _, part := range parts {
		if strings.Contains(part, "=") {
			kv := strings.SplitN(part, "=", 2)
			if len(kv) == 2 {
				key := strings.TrimSpace(kv[0])
				value := strings.TrimSpace(kv[1])
				if key != "" && value != "" {
					labels[key] = value
				}
			}
		}
	}
	return labels
}

// Describe 实现Prometheus的Collector接口
func (h *CustomHistogram) Describe(ch chan<- *prometheus.Desc) {
	// 注册 Desc
	ch <- h.desc
}

// TODO Collect方法中通过prometheus.NewDesc动态创建Desc实例，但未在Describe方法中预先注册。Prometheus要求所有Metric的Desc必须在Describe阶段注册，否则Collect阶段生成的Metric将被忽略。应将Desc实例在结构体中缓存，并在Describe和Collect中复用同一实例。
func (h *CustomHistogram) Collect(ch chan<- prometheus.Metric) {
	h.mu.Lock()
	defer h.mu.Unlock()

	for labelKey, data := range h.data {
		labels := h.parseLabelKey(labelKey)

		promMetric := prometheus.MustNewConstHistogram(
			h.desc,
			data.SampleCount,
			data.SampleSum,
			data.Buckets,
			h.parseLabelValue(labels)...,
		)

		ch <- promMetric
	}
}

func (h *CustomHistogram) parseLabelValue(labels prometheus.Labels) []string {
	values := make([]string, 0, len(h.labelNames))
	for _, name := range h.labelNames {
		if value, exists := labels[name]; exists {
			values = append(values, value)
		} else {
			values = append(values, "")
		}
	}
	return values
}