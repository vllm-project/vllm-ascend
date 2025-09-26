package logger

import (
	"fmt"
	"github.com/rs/zerolog/log"
)

type Log struct {
	Info   func(msg string)
	Infof  func(format string, args ...interface{})
	Error  func(msg string)
	Errorf func(format string, args ...interface{})
}

var logger *Log

func InitLogger() {
	if logger == nil {
		logger = &Log{
			Info:   Info,
			Infof:  Infof,
			Error:  Error,
			Errorf: Errorf,
		}
	}
}

func SetLogger(newLogger *Log) {
	logger = newLogger
}

func Logger() *Log {
	if logger == nil {
		InitLogger()
	}
	return logger
}

func Info(msg string) {
	log.Info().Msg(msg)
}

func Infof(format string, args ...interface{}) {
	log.Info().Msg(fmt.Sprintf(format, args...))
}

func Error(msg string) {
	log.Error().Msg(msg)
}

func Errorf(format string, args ...interface{}) {
	log.Error().Msg(fmt.Sprintf(format, args...))
}