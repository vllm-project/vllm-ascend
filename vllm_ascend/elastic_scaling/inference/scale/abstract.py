def reload_model(self):
    _ = self.collective_rpc("reload_model")
    return

def reload_kvcache(self):
    _ = self.collective_rpc("reload_kvcache")
    return