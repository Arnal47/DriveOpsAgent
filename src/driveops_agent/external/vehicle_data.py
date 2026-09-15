from .base import Adapter


class VehicleDataAdapter(Adapter):
    def list_logs(self):
        return self.request("list_logs", {})

    def read_log(self, log_id):
        return self.request("read_log", {"log_id": log_id})

    def query_signals(self, scenario_id):
        return self.request("query_signals", {"scenario_id": scenario_id})
