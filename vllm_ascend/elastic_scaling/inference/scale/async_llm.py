async def reload_models(self):
    await self.engine_core.reload_models_async()


async def reload_kvcache(self):
    await self.engine_core.reload_kvcache_async()
