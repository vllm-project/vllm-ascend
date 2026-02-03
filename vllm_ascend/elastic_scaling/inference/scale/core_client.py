async def reload_models_async(self):
    await self.call_utility_async("reload_model")
    return {"status", "done"}

async def reload_kvcache_async(self):
    await self.call_utility_async("reload_kvcache")
    return {"status", "done"}

