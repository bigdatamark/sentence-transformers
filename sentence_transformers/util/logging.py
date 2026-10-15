# [MODULE DOC]
# HELPER: Logging helpers: a handler that writes through tqdm so progress bars are not broken,
# and a function to install colored console logging with an extra NOTICE level.
from __future__ import annotations

import logging

import tqdm


class LoggingHandler(logging.Handler):
    # [CLASS DOC]
    # HELPER: logging.Handler subclass that formats each record and writes it using tqdm's write
    # function (so active progress bars are preserved).
    def __init__(self, level=logging.NOTSET) -> None:
        # [FUNCTION DOC]
        # HELPER: Initializes the handler with the given level (default NOTSET).
        super().__init__(level)

    def emit(self, record) -> None:
        # [FUNCTION DOC]
        # HELPER: Formats the record, writes it via tqdm write, and flushes. KeyboardInterrupt and
        # SystemExit propagate; any other exception is passed to the standard handleError.
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def install_logger(given_logger, level=logging.WARNING, fmt="%(levelname)s:%(name)s:%(message)s") -> None:
    # [FUNCTION DOC]
    # HELPER: Configures given_logger with coloredlogs using the given level and format string.
    # First registers a NOTICE level (25, between INFO and WARNING) and adds a "notice" method to all
    # Logger instances (global side effect). Uses default coloredlogs styles but with no asctime style,
    # faint white debug and bold cyan notice. Requires the optional coloredlogs package (ImportError
    # otherwise). Returns None.
    """Configures the given logger: format, logging level, style, etc"""
    import coloredlogs

    def add_notice_log_level():
        # [FUNCTION DOC]
        # HELPER: Registers level name "NOTICE" for number 25 and monkeypatches a notice method onto logging.Logger.
        """Creates a new 'notice' logging level"""
        # inspired by:
        # https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
        NOTICE_LEVEL_NUM = 25
        logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")

        def notice(self, message, *args, **kws):
            # [FUNCTION DOC]
            # HELPER: Logs message at NOTICE level if the logger is enabled for it, forwarding args and keyword options.
            if self.isEnabledFor(NOTICE_LEVEL_NUM):
                self._log(NOTICE_LEVEL_NUM, message, args, **kws)

        logging.Logger.notice = notice

    # Add an extra logging level above INFO and below WARNING
    add_notice_log_level()

    # More style info at:
    # https://coloredlogs.readthedocs.io/en/latest/api.html
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {"color": "white", "faint": True}
    level_styles["notice"] = {"color": "cyan", "bold": True}

    coloredlogs.install(
        logger=given_logger,
        level=level,
        use_chroot=False,
        fmt=fmt,
        level_styles=level_styles,
        field_styles=field_styles,
    )
