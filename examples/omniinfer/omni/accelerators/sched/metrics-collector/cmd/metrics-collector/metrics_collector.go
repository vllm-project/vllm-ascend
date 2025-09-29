package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"metrics-collector/pkg/metrics-collector"
	"metrics-collector/pkg/metrics-collector/logger"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
	"fmt"
	"path/filepath"
	"runtime"

	"github.com/gin-gonic/gin"
)

// 验证IP:端口,IP:端口格式的正则表达式
func isValidIPPortList(s string) bool {
	// 正则表达式说明：
	// 1. IP地址部分：支持IPv4或localhost
	//    - IPv4：四组0-255的数字，无前置零
	//    - localhost：直接匹配字符串"localhost"
	// 2. 端口部分：1-65535之间的整数
	// 3. 整体：以IP:端口开头，后续可跟逗号+IP:端口，允许0个或多个
	const pattern = `^(localhost|(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)):([1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])(,(localhost|(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)\.(25[0-5]|2[0-4]\d|1\d\d|[1-9][0-9]?|0)):([1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5]))*$`
	// 编译正则表达式
	re := regexp.MustCompile(pattern)
	return re.MatchString(s)
}

// 获取当前文件所在目录
func getCurrentFileDir() (string, error) {
	_, file, _, ok := runtime.Caller(1)
	if !ok {
		return "", fmt.Errorf("无法获取调用栈信息")
	}
	// 从文件路径中提取目录
	dir := filepath.Dir(file)
	return dir, nil
}

func main() {

	// 定义参数变量
	var (
		metricsServerIpAndPort = flag.String("metrics_collector_server", "",
			"metric collector server ip and port")
		schedulerServerIpAndPort = flag.String("scheduler_server", "",
			"scheduler server ip and port")
		prefillServersList = flag.String("prefill_servers_list", "",
			"prefill servers ip and port list, eg: ip1:port1,ip2:port2")
		decodeServersList = flag.String("decode_servers_list", "",
			"decode servers ip and port list, eg: ip1:port1,ip2:port2")
		metricsConfigYamlPath = flag.String("metrics_config_yaml_path", "../../deploy/metrics_config_vllm_0.9.0.yaml",
			"metrics config yaml path")
	)

	// 解析参数
	flag.Parse()
	// 检查合法性
	if !isValidIPPortList(*metricsServerIpAndPort) {
		logger.Logger().Errorf("metrics_collector_server 配置格式错误")
		os.Exit(1)
	}
	if !isValidIPPortList(*schedulerServerIpAndPort) {
		logger.Logger().Errorf("scheduler_server 配置格式错误")
		os.Exit(1)
	}
	if !isValidIPPortList(*prefillServersList) {
		logger.Logger().Errorf("prefill_servers_list 配置格式错误")
		os.Exit(1)
	}
	if !isValidIPPortList(*decodeServersList) {
		logger.Logger().Errorf("decode_servers_list 配置格式错误")
		os.Exit(1)
	}
	if *metricsConfigYamlPath == "../../deploy/metrics_config_vllm_0.9.0.yaml" {
		dir, err := getCurrentFileDir()
		if err != nil {
			logger.Logger().Errorf("获取默认metrics config yaml文件失败")
			os.Exit(1)
		}
		*metricsConfigYamlPath = filepath.Join(dir, *metricsConfigYamlPath)
	}

	instances, err := initInstance(*schedulerServerIpAndPort, *prefillServersList, *decodeServersList)
	if err != nil {
		os.Exit(1)
	}
	collectInterval := 5 * time.Second
	collector, err := metrics_collector.NewMetricsCollector(instances, *metricsConfigYamlPath)
	if err != nil {
		os.Exit(1)
	}
	engine := gin.New()
	engine.GET("/metrics", collector.HandleMetricsRequest)
	metricsServerPort := strings.Split(*metricsServerIpAndPort, ":")[1]
	server := &http.Server{
		Addr:    ":" + metricsServerPort,
		Handler: engine,
	}

	go func() {
		logger.Logger().Info("start http server: http://localhost:" + metricsServerPort)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Logger().Errorf("http server error: %v", err.Error())
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), collectInterval)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("server shutdown error: %v", err)
	}

	logger.Logger().Info("server shutdown successfully")
}

func initInstance(schedulerServerIpAndPort string, prefillServersList string, decodeServersList string) ([]metrics_collector.Instance, error) {
	instances := make([]metrics_collector.Instance, 0)

	schedulerServerIp := strings.Split(schedulerServerIpAndPort, ":")[0]
	schedulerServerPortString := strings.Split(schedulerServerIpAndPort, ":")[1]
	schedulerServerPort, err := strconv.Atoi(schedulerServerPortString)
	if err != nil {
		logger.Logger().Errorf("scheduler server端口转换整型数据类型失败（%s）: %v\n", schedulerServerPortString, err.Error())
		return nil, err
	}
	instances = append(instances, metrics_collector.Instance{
		Role: "Scheduler",
		IP:   schedulerServerIp,
		Port: schedulerServerPort,
	})

	prefillServers := strings.Split(prefillServersList, ",")
	for _, prefillNodeInfo := range prefillServers {
		prefillIpPort := strings.Split(prefillNodeInfo, ":")
		prefillPort, err := strconv.Atoi(prefillIpPort[1])
		if err != nil {
			logger.Logger().Errorf("prefill端口转换整型数据类型失败（%s）: %v\n", prefillIpPort[1], err.Error())
			return nil, err
		}
		instances = append(instances, metrics_collector.Instance{
			Role: "Prefill",
			IP:   prefillIpPort[0],
			Port: prefillPort,
		})

	}

	decodeServers := strings.Split(decodeServersList, ",")
	for _, decodeNodeInfo := range decodeServers {
		decodeIpPort := strings.Split(decodeNodeInfo, ":")
		decodePort, err := strconv.Atoi(decodeIpPort[1])
		if err != nil {
			logger.Logger().Errorf("decode端口转换整型数据类型失败（%s）: %v\n", decodeIpPort[1], err.Error())
			return nil, err
		}
		instances = append(instances, metrics_collector.Instance{
			Role: "decode",
			IP:   decodeIpPort[0],
			Port: decodePort,
		})
	}

	logger.Logger().Infof("%v", instances)
	return instances, nil
}
