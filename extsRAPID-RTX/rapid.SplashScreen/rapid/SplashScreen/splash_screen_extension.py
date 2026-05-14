import asyncio
import omni.ext
import omni.kit.app
import omni.splash


class SplashScreenExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[rapid.SplashScree] Startup...")

        # 启动异步任务显示并管理闪屏
        self._splash_task = asyncio.ensure_future(self._handle_splash_screen())

    async def _handle_splash_screen(self):

        # Close the splash image
        for _ in range(50):
            await omni.kit.app.get_app().next_update_async()
        omni.splash.acquire_splash_screen_interface().close_all()

    def on_shutdown(self):
        # 如果闪屏还没关掉，强制清理
        if self._splash_task:
            self._splash_task.cancel()
        print("[rapid.SplashScree] Shutdown.")
