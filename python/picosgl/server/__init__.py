def launch_server() -> None:
    from .launch import launch_server as _launch_server

    _launch_server()

__all__ = ["launch_server"]
