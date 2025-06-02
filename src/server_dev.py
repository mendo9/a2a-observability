#!/usr/bin/env python3
"""
Development server with hot reload functionality
"""

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import the main server
from server import main


class HotReloadHandler(FileSystemEventHandler):
    """File system event handler for hot reload"""

    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_reload = 0

    def on_modified(self, event):
        if event.is_directory:
            return

        # Only react to Python files
        if not event.src_path.endswith(".py"):
            return

        # Debounce rapid file changes
        current_time = time.time()
        if current_time - self.last_reload < 1.0:  # 1 second debounce
            return

        self.last_reload = current_time
        print(f"🔄 File changed: {event.src_path}")
        print("🚀 Restarting server...")
        self.restart_callback()


class DevelopmentServer:
    """Development server with hot reload"""

    def __init__(self):
        self.server_task: Optional[asyncio.Task] = None
        self.observer: Optional[Observer] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.should_restart = False

    async def start_server(self):
        """Start the main server"""
        try:
            await main()
        except KeyboardInterrupt:
            print("🛑 Server interrupted")
        except Exception as e:
            print(f"❌ Server error: {e}")

    def restart_server(self):
        """Signal that server should restart"""
        self.should_restart = True
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()

    def setup_file_watcher(self):
        """Setup file system watcher for hot reload"""
        watch_dir = Path("/app/src")

        if not watch_dir.exists():
            print(f"⚠️  Watch directory {watch_dir} does not exist")
            return

        self.observer = Observer()
        handler = HotReloadHandler(self.restart_server)
        self.observer.schedule(handler, str(watch_dir), recursive=True)
        self.observer.start()
        print(f"👀 Watching {watch_dir} for changes...")

    async def run(self):
        """Run the development server with hot reload"""
        print("🔧 Starting Development Server with Hot Reload")
        print("=" * 50)

        # Setup file watcher
        self.setup_file_watcher()

        try:
            while True:
                self.should_restart = False

                # Start the server
                print("🚀 Starting server...")
                self.server_task = asyncio.create_task(self.start_server())

                try:
                    await self.server_task
                    # If we get here, server exited normally
                    break
                except asyncio.CancelledError:
                    if self.should_restart:
                        print("🔄 Restarting due to file changes...")
                        # Small delay to let file operations complete
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        # Manual cancellation
                        break

        except KeyboardInterrupt:
            print("\n👋 Development server stopped by user")
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()

            if self.server_task and not self.server_task.done():
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run development server
    dev_server = DevelopmentServer()
    asyncio.run(dev_server.run())
