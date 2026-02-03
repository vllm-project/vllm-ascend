import os
def reload_model(self):
    ## Call load_model() with updated INFER_STATUS env var
    os.environ["INFER_STATUS"] = "1"
    self.model_runner.load_model()