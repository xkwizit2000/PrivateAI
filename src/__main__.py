from .config import CHAT_BACKEND


def main() -> None:
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    backend = CHAT_BACKEND
    if backend == "session":
        from .bridge import main as bridge_main

        bridge_main()
        return
    if backend in {"telegram", ""}:
        from .bot import main as telegram_main

        telegram_main()
        return
    raise ValueError(
        f"Unsupported CHAT_BACKEND={backend!r}. Use 'telegram' or 'session'."
    )


if __name__ == "__main__":
    main()
