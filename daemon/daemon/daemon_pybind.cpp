#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inc/daemon.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mdaemon_py, m) {
    m.doc() = "Python bindings for the multimodel daemon (handle pools + model management)";

    py::enum_<mdaemon::ModelState>(m, "ModelState")
        .value("UNINITIALIZED", mdaemon::ModelState::UNINITIALIZED)
        .value("INITIALIZED", mdaemon::ModelState::INITIALIZED)
        .value("REGISTERED", mdaemon::ModelState::REGISTERED)
        .value("DONE", mdaemon::ModelState::DONE)
        .value("INVALID", mdaemon::ModelState::INVALID)
        .export_values();

    py::enum_<mdaemon::MessageState>(m, "MessageState")
        .value("NONE", mdaemon::MessageState::NONE)
        .value("REQUEST_REGISTER", mdaemon::MessageState::REQUEST_REGISTER)
        .value("REGISTERED", mdaemon::MessageState::REGISTERED)
        .value("REQUEST_GET_HANDLES", mdaemon::MessageState::REQUEST_GET_HANDLES)
        .value("HANDLES_READY", mdaemon::MessageState::HANDLES_READY)
        .value("REQUEST_RETURN_HANDLES", mdaemon::MessageState::REQUEST_RETURN_HANDLES)
        .value("HANDLES_RETURNED", mdaemon::MessageState::HANDLES_RETURNED)
        .value("DONE", mdaemon::MessageState::DONE)
        .value("INVALID", mdaemon::MessageState::INVALID)
        .export_values();

    py::class_<mdaemon::Daemon::HandlePoolDeviceSnapshot>(m, "HandlePoolDeviceSnapshot")
        .def_readonly("granularity", &mdaemon::Daemon::HandlePoolDeviceSnapshot::granularity)
        .def_readonly("device_id", &mdaemon::Daemon::HandlePoolDeviceSnapshot::device_id)
        .def_readonly("total_handles", &mdaemon::Daemon::HandlePoolDeviceSnapshot::total_handles)
        .def_readonly("used_handles", &mdaemon::Daemon::HandlePoolDeviceSnapshot::used_handles)
        .def_readonly("available_handles", &mdaemon::Daemon::HandlePoolDeviceSnapshot::available_handles)
        .def_readonly("total_bytes", &mdaemon::Daemon::HandlePoolDeviceSnapshot::total_bytes)
        .def_readonly("used_bytes", &mdaemon::Daemon::HandlePoolDeviceSnapshot::used_bytes)
        .def_readonly("available_bytes", &mdaemon::Daemon::HandlePoolDeviceSnapshot::available_bytes);

    py::class_<mdaemon::Daemon::ModelSnapshot>(m, "ModelSnapshot")
        .def_readonly("model_id", &mdaemon::Daemon::ModelSnapshot::model_id)
        .def_readonly("state", &mdaemon::Daemon::ModelSnapshot::state)
        .def_readonly("message_state", &mdaemon::Daemon::ModelSnapshot::message_state)
        .def_readonly("model_npuid", &mdaemon::Daemon::ModelSnapshot::model_npuid)
        .def_readonly("model_osid", &mdaemon::Daemon::ModelSnapshot::model_osid)
        .def_readonly("allocated_bytes", &mdaemon::Daemon::ModelSnapshot::allocated_bytes)
        .def_readonly("allocated_handles", &mdaemon::Daemon::ModelSnapshot::allocated_handles);

    py::class_<mdaemon::Daemon>(m, "Daemon")
        .def(py::init<>())
        .def("start", &mdaemon::Daemon::start)
        .def("stop", &mdaemon::Daemon::stop)
        .def("is_running", &mdaemon::Daemon::isRunning)
        // .def("run_once", &mdaemon::Daemon::runOnce)
        // .def("register_model_from_macro",
        //      [](mdaemon::Daemon& d) -> py::object {
        //          mdaemon::ModelConfig* cfg = d.registerModelFromMacro();
        //          if (!cfg) {
        //              return py::none();
        //          }
        //          return py::int_(cfg->getModelId());
        //      })
        .def("get_model_snapshot", &mdaemon::Daemon::getModelSnapshot,
             py::arg("model_id"), py::arg("sync_message_space_from_shm") = false)
        .def("get_model_state", &mdaemon::Daemon::getModelState)
        // .def("update_model_state", &mdaemon::Daemon::updateModelState)
        // .def("list_model_ids", &mdaemon::Daemon::listModelIds)
        // .def("get_current_model_npuids", &mdaemon::Daemon::getCurrentModelNpuids)
        .def("initialize_handle_pool_device", &mdaemon::Daemon::initializeHandlePoolDevice,
             py::arg("granularity"), py::arg("device_id"), py::arg("total_bytes"))
        .def("handle_pool_available", &mdaemon::Daemon::handlePoolAvailable,
             py::arg("granularity"), py::arg("device_id"))
        .def("extend_handles", &mdaemon::Daemon::extendHandles,
             py::arg("granularity"), py::arg("device_id"), py::arg("count"))
        .def("remove_handles", &mdaemon::Daemon::removeHandles,
             py::arg("granularity"), py::arg("device_id"), py::arg("count"))
        .def("snapshot_handle_pools", &mdaemon::Daemon::snapshotHandlePools)
        .def("snapshot_models", &mdaemon::Daemon::snapshotModels,
             py::arg("sync_message_space_from_shm") = false)
        .def("drain_logs", &mdaemon::Daemon::drainLogs);
}
