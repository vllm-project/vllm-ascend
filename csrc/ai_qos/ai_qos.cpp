#include <pybind11/pybind11.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include "dsmi_common_interface.h"

#define DSMI_QOS_INDEX_OFFSET 8
#define DSMI_QOS_INDEX_LEN 8U
#define DSMI_QOS_MAIN_INDEX_OFFSET 8U
#define DSMI_QOS_SUB_INDEX_OFFSET 16U
#define DSMI_QOS_THIRD_INDEX_OFFSET 24U

#define PCIEDMA_MASTER 7
#define SDMA_MASTER 13

#define DSMI_QOS_SUB_CMD_MAKE(qos_index, qos_sub_cmd) (((qos_index) << DSMI_QOS_INDEX_OFFSET) | (qos_sub_cmd))
#define DSMI_QOS_SUB_CMD_MAKE_V2(qos_main_index, qos_sub_index, qos_third_index, qos_sub_cmd) \
  ((((qos_main_index) & ((1U << DSMI_QOS_INDEX_LEN) - 1U)) << DSMI_QOS_MAIN_INDEX_OFFSET) |   \
   (((qos_sub_index) & ((1U << DSMI_QOS_INDEX_LEN) - 1U)) << DSMI_QOS_SUB_INDEX_OFFSET) |     \
   (((qos_third_index) & ((1U << DSMI_QOS_INDEX_LEN) - 1U)) << DSMI_QOS_THIRD_INDEX_OFFSET) | (qos_sub_cmd))

int set_fuse_gbl_config(unsigned int device_id, uint32_t enable, uint32_t autoqos_fuse_en, int mpamqos_fuse_mode) {
  struct qos_gbl_config gblCfg = {0};
  gblCfg.enable = enable;
  gblCfg.autoqos_fuse_en = autoqos_fuse_en;
  gblCfg.mpamqos_fuse_mode = mpamqos_fuse_mode;
  int ret = dsmi_set_device_info(device_id, DSMI_MAIN_CMD_QOS, static_cast<uint32_t>(DSMI_QOS_SUB_GLOBAL_CONFIG),
                                 static_cast<void*>(&gblCfg), sizeof(struct qos_gbl_config));
  if (ret != 0) {
    printf("[dev:%d] set fuse gbl (en=%u auto=%u mode=%d) failed, ret = %d\n", device_id, enable, autoqos_fuse_en,
           mpamqos_fuse_mode, ret);
    return ret;
  }
  return ret;
}

int set_qos(unsigned int device_id, int master, int mpamid, int qos, int pmg, int mode) {
  struct qos_master_config masterCfg = {0};
  masterCfg.master = master;
  masterCfg.mpamid = mpamid;
  masterCfg.qos = qos;
  masterCfg.pmg = pmg;
  if (master == PCIEDMA_MASTER) {
    masterCfg.bitmap[0] = 0x1;
  } else if (master == SDMA_MASTER) {
    masterCfg.bitmap[0] = 0xffffffffffffffff;
  }
  masterCfg.mode = mode;
  int ret = dsmi_set_device_info(device_id, DSMI_MAIN_CMD_QOS, static_cast<uint32_t>(DSMI_QOS_SUB_MASTER_CONFIG),
                                 static_cast<void*>(&masterCfg), sizeof(struct qos_master_config));
  if (ret != 0) {
    printf("[dev:%d] set qos = %d failed, ret = %d\n", device_id, qos, ret);
    return ret;
  }
  return ret;
}

int set_bw(unsigned int device_id, int mpamid, int bw_low, int bw_high, int hardlimit) {
  struct qos_mata_config mataCfg = {0};
  mataCfg.mpamid = mpamid;
  mataCfg.bw_low = bw_low;
  mataCfg.bw_high = bw_high;
  mataCfg.hardlimit = hardlimit;
  int ret = dsmi_set_device_info(device_id, DSMI_MAIN_CMD_QOS, static_cast<uint32_t>(DSMI_QOS_SUB_MATA_CONFIG),
                                 static_cast<void*>(&mataCfg), sizeof(struct qos_mata_config));
  if (ret != 0) {
    printf("[dev:%d] mpamid: %d set bw: %d-%d failed, ret = %d\n", device_id, mpamid, bw_low, bw_high, ret);
    return ret;
  }
  return ret;
}

std::tuple<int, unsigned int, unsigned int, int> get_bw(unsigned int device_id, int mpamid) {
  struct qos_mata_config mataCfg = {0};
  mataCfg.mpamid = mpamid;
  uint32_t size = sizeof(struct qos_mata_config);
  uint32_t subCmd = static_cast<uint32_t>(DSMI_QOS_SUB_CMD_MAKE(mataCfg.mpamid, DSMI_QOS_SUB_MATA_CONFIG));
  int ret = dsmi_get_device_info(device_id, DSMI_MAIN_CMD_QOS, subCmd, static_cast<void*>(&mataCfg), &size);
  if (ret != 0 || size != sizeof(struct qos_mata_config)) {
    printf("[dev:%d] mpamid: %d get bw failed, ret = %d, size = %u, main cmd = %#x, sub cmd = %#x\n", device_id, mpamid,
           ret, size, DSMI_MAIN_CMD_QOS, subCmd);
    int err = (ret != 0) ? ret : -1;
    return std::make_tuple(err, 0, 0, 0);
  }
  return std::make_tuple(ret, mataCfg.bw_low, mataCfg.bw_high, mataCfg.hardlimit);
}

std::tuple<int, int, int, int, int, unsigned int> get_qos(unsigned int device_id, int master) {
  struct qos_master_config masterCfg = {0};
  masterCfg.master = master;
  uint32_t size = sizeof(struct qos_master_config);
  int coreid = 0;
  uint32_t subCmd = static_cast<uint32_t>(
      DSMI_QOS_SUB_CMD_MAKE_V2(static_cast<uint32_t>(masterCfg.master), coreid, 0, DSMI_QOS_SUB_MASTER_CONFIG));
  int ret = dsmi_get_device_info(device_id, DSMI_MAIN_CMD_QOS, subCmd, static_cast<void*>(&masterCfg), &size);
  if (ret != 0 || size != sizeof(struct qos_master_config)) {
    printf("[dev:%d] get qos failed, ret = %d, size = %u, main cmd = %#x, sub cmd = %#x\n", device_id, ret, size,
           DSMI_MAIN_CMD_QOS, subCmd);
    int err = (ret != 0) ? ret : -1;
    return std::make_tuple(err, 0, 0, 0, 0, 0U);
  }
  return std::make_tuple(0, static_cast<int>(masterCfg.master), static_cast<int>(masterCfg.mpamid),
                         static_cast<int>(masterCfg.qos), static_cast<int>(masterCfg.pmg), masterCfg.mode);
}

std::tuple<int, unsigned int, unsigned int, unsigned int> get_fuse_mode(unsigned int device_id) {
  struct qos_gbl_config gblCfg = {0};
  uint32_t size = sizeof(struct qos_gbl_config);
  int ret = dsmi_get_device_info(device_id, DSMI_MAIN_CMD_QOS, static_cast<uint32_t>(DSMI_QOS_SUB_GLOBAL_CONFIG),
                                 static_cast<void*>(&gblCfg), &size);
  if (ret != 0 || size != sizeof(struct qos_gbl_config)) {
    printf("[dev:%d] get fuse mode failed, ret = %d, size = %u, main cmd = %#x, sub cmd = %#x\n", device_id, ret, size,
           DSMI_MAIN_CMD_QOS, DSMI_QOS_SUB_GLOBAL_CONFIG);
    int err = (ret != 0) ? ret : -1;
    return std::make_tuple(err, 0U, 0U, 0U);
  }
  return std::make_tuple(0, gblCfg.enable, gblCfg.autoqos_fuse_en, gblCfg.mpamqos_fuse_mode);
}

namespace py = pybind11;
PYBIND11_MODULE(ai_qos, m) {
  m.doc() = "AI QoS(Quality of Service) control module for hardware resource management";
  m.def("set_qos", &set_qos, py::arg("device_id"), py::arg("master"), py::arg("mpamid"), py::arg("qos"), py::arg("pmg"),
        py::arg("mode"));
  m.def("set_fuse_gbl_config", &set_fuse_gbl_config, py::arg("device_id"), py::arg("enable"),
        py::arg("autoqos_fuse_en"), py::arg("mpamqos_fuse_mode"));
  m.def("get_qos", &get_qos,
        "Returns (ret, master, mpamid, qos, pmg, mode). ret==0 on success; on failure ret is DSMI error (or -1 if size "
        "mismatch).",
        py::arg("device_id"), py::arg("master"));
  m.def("set_bw", &set_bw, py::arg("device_id"), py::arg("mpamid"), py::arg("bw_low"), py::arg("bw_high"),
        py::arg("hardlimit"));
  m.def("get_bw", &get_bw, py::arg("device_id"), py::arg("mpamid"));
  m.def("get_fuse_mode", &get_fuse_mode, py::arg("device_id"));
}
