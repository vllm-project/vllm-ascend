package metrics_collector

import (
	"fmt"
	"io"
	"math"
	"metrics-collector/pkg/metrics-collector/logger"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
)

var metricChan chan struct{}

type Instance struct {
	Role          string
	IP            string
	Port          int
	MetricsLabels map[string]map[string]prometheus.Labels
}

// 更新instance已经收集的标签信息
func (i *Instance) updateInstanceLabels(name string, labels prometheus.Labels) {
	// 如果该指标没有被记录，初始化一个
	if i.MetricsLabels[name] == nil {
		i.MetricsLabels[name] = make(map[string]prometheus.Labels)
	}

	// 生成标签，快速去重
	labelKey := generateLabelKey(labels)

	// 检查标签是否存在，避免重复
	if _, exists := i.MetricsLabels[name][labelKey]; exists {
		return
	}

	i.MetricsLabels[name][labelKey] = labels
}

// 生成标签的唯一键，用于快速去重
func generateLabelKey(labels prometheus.Labels) string {
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// 构建字符串
	var builder strings.Builder
	for i, k := range keys {
		if i > 0 {
			builder.WriteString("|")
		}
		builder.WriteString(k)
		builder.WriteString("=")
		builder.WriteString(labels[k])
	}
	return builder.String()
}

// 获取指定指标在该instances上的所有标签信息
func (i *Instance) getInstanceLabelsForMetrics(metricsName string) []prometheus.Labels {
	var allLabels []prometheus.Labels
	if labels, exists := i.MetricsLabels[metricsName]; exists {
		for _, label := range labels {
			allLabels = append(allLabels, label)
		}
	}
	return allLabels
}

type Collector struct {
	mapMutex                   sync.RWMutex
	collectEnable              bool
	instances                  []Instance
	registry                   *prometheus.Registry
	aggRegistry                *prometheus.Registry
	interval                   time.Duration
	gaugeMetrics               map[string]*prometheus.GaugeVec
	counterMetrics             map[string]*prometheus.CounterVec
	histogramMetrics           map[string]*CustomHistogram
	aggregatedGaugeMetrics     map[string]*prometheus.GaugeVec
	aggregatedCounterMetrics   map[string]*prometheus.CounterVec
	aggregatedHistogramMetrics map[string]*CustomHistogram
	metricsConfig              MetricsConfig
}

var isRunning func() bool

func defaultRunning() bool {
	return true
}

func InitRunning() {
	if isRunning == nil {
		isRunning = defaultRunning
	}
}

func SetRunningFun(customizeRunning func() bool) {
	if isRunning == nil {
		logger.Logger().Infof("customize running function")
		isRunning = customizeRunning
	}
}

func NewMetricsCollector(instances []Instance, yamlPath string) (*Collector, error) {
	logger.InitLogger()
	InitRunning()
	registry := prometheus.NewRegistry()
	aggRegistry := prometheus.NewRegistry()

	metricsConfig, err := loadMetricsYamlConfig(yamlPath)
	if err != nil {
		logger.Logger().Errorf("error when load metrics yaml config %s", err.Error())
		return &Collector{
			collectEnable: false,
			registry:      registry,
			aggRegistry:   aggRegistry,
		}, err
	}

	// 各个实例指标初始化
	gaugeMetrics := make(map[string]*prometheus.GaugeVec)
	counterMetrics := make(map[string]*prometheus.CounterVec)
	histogramMetrics := make(map[string]*CustomHistogram)

	// 聚合指标初始化
	aggregatedGaugeMetrics := make(map[string]*prometheus.GaugeVec)
	aggregatedCounterMetrics := make(map[string]*prometheus.CounterVec)
	aggregatedHistogramMetrics := make(map[string]*CustomHistogram)

	for i := range instances {
		instances[i].MetricsLabels = make(map[string]map[string]prometheus.Labels)
	}

	metricChan = make(chan struct{})
	interval := metricsConfig.MetricsCollectInterval

	metricsCollector := &Collector{
		mapMutex:      sync.RWMutex{},
		collectEnable: true,
		// 所有的实例
		instances: instances,
		// 两个注册表，保证可见性
		registry:    registry,
		aggRegistry: aggRegistry,
		// 时间间隔
		interval: time.Duration(interval) * time.Second,
		// 原始数据
		gaugeMetrics:     gaugeMetrics,
		counterMetrics:   counterMetrics,
		histogramMetrics: histogramMetrics,
		// 聚合数据
		aggregatedGaugeMetrics:     aggregatedGaugeMetrics,
		aggregatedCounterMetrics:   aggregatedCounterMetrics,
		aggregatedHistogramMetrics: aggregatedHistogramMetrics,
		// 汇聚配置
		metricsConfig: *metricsConfig,
	}
	logger.Logger().Infof("init metrics collector successful and interval time is %d seconds", interval)
	metricsCollector.runMetricsCollectLoop()
	return metricsCollector, nil
}

func (c *Collector) HandleMetricsRequest(ctx *gin.Context) {
	// 检查URL参数
	detailEnabled := ctx.Query("detailEnable")
	var registry *prometheus.Registry

	if strings.ToLower(detailEnabled) == "true" {
		// 返回原始指标
		registry = c.registry
	} else {
		// 返回聚合指标
		registry = c.aggRegistry
	}

	// 使用prometheus的HTTP处理器输出指标
	handler := promhttp.HandlerFor(registry, promhttp.HandlerOpts{})
	handler.ServeHTTP(ctx.Writer, ctx.Request)
}

func (c *Collector) runMetricsCollectLoop() {
	if !c.collectEnable {
		logger.Logger().Errorf("fail to init metrics collector")
		return
	}
	// 第一次触发串行执行，收集所有的指标信息
	if isRunning() {
		c.collectMetrics()
	}

	go func() {
		logger.Logger().Infof("start server to collect metrics")
		ticker := time.NewTicker(c.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if isRunning() {
					c.collectMetrics()
				}
			case <-metricChan:
				logger.Logger().Infof("metrics collector stop")
				return
			}
		}
	}()
}

func (c *Collector) collectMetrics() {
	var wg sync.WaitGroup
	for _, instance := range c.instances {
		wg.Add(1)
		// 为每个实例启动一个goroutine进行并发
		go func(instance Instance) {
			defer wg.Done()
			metrics, err := c.fetchMetrics(instance)
			if err != nil {
				logger.Logger().Errorf("failed to fetch metrics: %s:%d, role %s error %s", instance.IP, instance.Port, instance.Role, err.Error())
				return
			}
			c.recordMetrics(&instance, metrics)
		}(instance)
	}

	// 等待所有指标收集完成
	wg.Wait()

	// 执行指标汇聚
	c.AggregateMetrics()
}

func (c *Collector) fetchMetrics(instance Instance) (string, error) {
	// 设置获取metrics指标请求超时时间
	client := &http.Client{Timeout: time.Duration(c.metricsConfig.MetricsRequestTimeout) * time.Second}

	// 根据instance的ip和port拼接请求
	var url string
	if instance.Role == "Scheduler" {
		url = fmt.Sprintf("http://%s:%d/internal/metrics", instance.IP, instance.Port)
	} else {
		url = fmt.Sprintf("http://%s:%d/metrics", instance.IP, instance.Port)
	}

	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to fetch metrics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	return string(body), nil
}

func (c *Collector) recordMetrics(instance *Instance, metrics string) {
	// 创建 Prometheus 文本格式的解析器
	parser := expfmt.TextParser{}
	reader := strings.NewReader(metrics)

	// 解析指标
	metricFamilies, err := parser.TextToMetricFamilies(reader)
	if err != nil {
		logger.Logger().Infof("failed to parse metrics: %v", err)
		return
	}

	// 这里可以优化逻辑，从我们需要的指标筛选
	for name, m := range metricFamilies {
		for _, metric := range m.GetMetric() {
			// 新添加label标签，当前是两个：
			// 1. role 角色： Prefill， Decode
			// 2. instance 当前默认是实例的ip
			labels := prometheus.Labels{
				"role":     instance.Role,
				"instance": fmt.Sprintf("%s:%d", instance.IP, instance.Port),
			}

			// 添加指标中已有的标签
			var list []string
			for _, label := range metric.GetLabel() {
				list = append(list, *label.Name)
				labels[*label.Name] = *label.Value
			}

			if strings.ToLower(instance.Role) == "scheduler" {
				vllmName := getRealMetricsName(name)
				if vllmName == "" {
					continue
				}
				name = vllmName
			}
			c.registerDiscoveredMetrics(m, list, name)
			switch m.GetType() {
			case dto.MetricType_GAUGE:
				recordGaugeMetrics(c, name, labels, metric)
			case dto.MetricType_COUNTER:
				recordCounterMetrics(c, name, labels, metric)
			case dto.MetricType_HISTOGRAM:
				recordHistogramMetrics(c, name, labels, metric)
			}
			// 维护指标中已收集的的标签信息
			instance.updateInstanceLabels(name, labels)
		}
	}
}

func (c *Collector) AggregateMetrics() {
	// 汇聚Gauge指标
	for _, cfg := range c.metricsConfig.Configurations {
		err := processMetricsAccordingConfiguration(cfg, c)
		if err != nil {
			logger.Logger().Errorf("Error processing configuration %s: %v", cfg.MetricsName, err)
			continue
		}
	}

}

func (c *Collector) isMetricRegistered(metricsName string) bool {
	if _, ok := c.gaugeMetrics[metricsName]; ok {
		return true
	}
	if _, ok := c.counterMetrics[metricsName]; ok {
		return true
	}
	if _, ok := c.histogramMetrics[metricsName]; ok {
		return true
	}
	return false
}

// 自动注册
func (c *Collector) registerDiscoveredMetrics(m *dto.MetricFamily, labels []string, metricsName string) {
	// 指标一级map只有在这里会被写，其他地方都是读。 二级map例如*prometheus.GaugeVec内部已有加锁
	c.mapMutex.Lock()
	defer c.mapMutex.Unlock()
	// 检查指标是否已经注册
	if c.isMetricRegistered(metricsName) {
		return
	}
	switch m.GetType() {
	case dto.MetricType_GAUGE:
		registerGaugeMetrics(c, m, labels, metricsName)
	case dto.MetricType_COUNTER:
		registerCounterMetrics(c, m, labels, metricsName)
	case dto.MetricType_HISTOGRAM:
		registerHistogramMetrics(c, m, labels, metricsName)
	}
}

func recordGaugeMetrics(c *Collector, name string, labels prometheus.Labels, metric *dto.Metric) {
	if gauge, ok := c.gaugeMetrics[name]; ok && metric.Gauge != nil {
		gauge.With(labels).Set(metric.GetGauge().GetValue())
	}
}

func recordHistogramMetrics(c *Collector, name string, labels prometheus.Labels, metric *dto.Metric) {
	if metric.Histogram != nil && c.histogramMetrics[name] != nil {
		// 原有histogram不支持直接赋值,需要自定义一个histogram来计算
		his := metric.Histogram
		sampleCount := his.GetSampleCount()
		sampleSum := his.GetSampleSum()
		buckets := his.GetBucket()

		hisData := &HistogramData{
			SampleCount: sampleCount,
			SampleSum:   sampleSum,
			Buckets:     make(map[float64]uint64),
		}
		for _, bucket := range buckets {
			hisData.Buckets[bucket.GetUpperBound()] = bucket.GetCumulativeCount()
		}

		if customHist, ok := c.histogramMetrics[name]; ok && metric.Histogram != nil {
			customHist.SetHistogramData(labels, hisData)
		}
	}
}

func registerGaugeMetrics(c *Collector, m *dto.MetricFamily, labels []string, metricsName string) {
	registerLabels := getCollectLabels()
	registerLabels = append(registerLabels, labels...)
	gauge := createNewGaugeVec(metricsName, *m.Help, registerLabels)
	err := c.registry.Register(gauge)
	if err != nil {
		logger.Logger().Errorf("fail to register metrics %s, error %s", metricsName, err.Error())
		return
	}
	// 成功注册后，更新map
	c.gaugeMetrics[metricsName] = gauge
	// 判断是否为Union指标
	isUnionMetrics := registerUnionMetrics(c, metricsName, gauge)
	if isUnionMetrics {
		return
	}
	// 注册需要聚合的指标
	aggGauge := createNewGaugeVec(metricsName, *m.Help, labels)
	err = c.aggRegistry.Register(aggGauge)
	if err != nil {
		logger.Logger().Errorf("fail to register aggregated metrics %s, error %s", metricsName, err.Error())
		return
	}
	// 成功注册后，更新map
	c.aggregatedGaugeMetrics[metricsName] = aggGauge
}

func registerCounterMetrics(c *Collector, m *dto.MetricFamily, labels []string, metricsName string) {
	registerLabels := getCollectLabels()
	registerLabels = append(registerLabels, labels...)
	counter := createNewCounterVec(metricsName, *m.Help, registerLabels)
	err := c.registry.Register(counter)
	if err != nil {
		logger.Logger().Errorf("fail to register metrics %s, error %s", metricsName, err.Error())
		return
	}
	c.counterMetrics[metricsName] = counter
	// 判断是否为Union指标
	isUnionMetrics := registerUnionMetrics(c, metricsName, counter)
	if isUnionMetrics {
		return
	}

	aggCounter := createNewCounterVec(metricsName, *m.Help, labels)
	err = c.aggRegistry.Register(aggCounter)
	if err != nil {
		logger.Logger().Errorf("fail to register aggregated metrics %s, error %s", metricsName, err.Error())
		return
	}
	c.aggregatedCounterMetrics[metricsName] = aggCounter
}

func registerHistogramMetrics(c *Collector, m *dto.MetricFamily, labels []string, metricsName string) {
	registerLabels := getCollectLabels()
	registerLabels = append(registerLabels, labels...)
	var buckets []float64
	for _, metric := range m.Metric {
		if metric.Histogram != nil {
			for _, bucket := range metric.Histogram.GetBucket() {
				upperBound := bucket.GetUpperBound()
				// 过滤+Inf边界，因为他不是真的bucket边界
				if !math.IsInf(upperBound, 1) {
					buckets = append(buckets, upperBound)
				}
			}
			// 只处理第一个指标实例的bucket信息
			break
		}
	}
	histogram := createHistogramVec(metricsName, *m.Help, registerLabels)
	err := c.registry.Register(histogram)
	if err != nil {
		logger.Logger().Errorf("fail to register metrics %s, error %s", metricsName, err.Error())
		return
	}
	c.histogramMetrics[metricsName] = histogram
	// 判断是否为Union指标
	isUnionMetrics := registerUnionMetrics(c, metricsName, histogram)
	if isUnionMetrics {
		return
	}

	aggHistogram := createHistogramVec(metricsName, *m.Help, labels)
	err = c.aggRegistry.Register(aggHistogram)
	if err != nil {
		logger.Logger().Errorf("fail to register aggregated metrics %s error %s", metricsName, err.Error())
		return
	}
	c.aggregatedHistogramMetrics[metricsName] = aggHistogram
}

func registerUnionMetrics(c *Collector, metricsName string, metrics prometheus.Collector) bool {
	if op, ok := c.metricsConfig.metricOperation[metricsName]; ok {
		// 对于组合的metrics指标，不需要创建聚合指标对象， 复用从各个实例获取的指标列表
		if op == "union" {
			err := c.aggRegistry.Register(metrics)
			if err != nil {
				logger.Logger().Errorf("fail to register union metrics %s, error %s", metricsName, err.Error())
				return true
			}
			return true
		}
	}
	return false
}

func createNewGaugeVec(metricsName string, help string, labels []string) *prometheus.GaugeVec {
	return prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: metricsName,
			Help: help,
		},
		labels,
	)
}

func createNewCounterVec(metricsName string, help string, labels []string) *prometheus.CounterVec {
	return prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: metricsName,
			Help: help,
		},
		labels,
	)
}

func createHistogramVec(metricsName string, help string, labels []string) *CustomHistogram {
	return NewCustomHistogram(
		prometheus.HistogramOpts{
			Name: metricsName,
			Help: help,
		}, labels,
	)
}

func StopMetricsCollector() {
	if metricChan != nil {
		logger.Logger().Infof("Stop metrics collector")
		close(metricChan)
	}
}
