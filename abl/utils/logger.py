# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
import sys
from logging import Logger, LogRecord
from typing import Optional, Union

from termcolor import colored

from .manager import ManagerMixin, _accquire_lock, _release_lock


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: str = "abl"):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class ABLFormatter(logging.Formatter):
    """Colorful format for ABLLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
        blink (bool): Whether to blink the ``INFO`` and ``DEBUG`` logging
            level.
        **kwargs: Keyword arguments passed to
            :meth:`logging.Formatter.__init__`.
    """

    _color_mapping: dict = dict(
        ERROR="red", WARNING="yellow", INFO="white", DEBUG="green"
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (
            not color and blink
        ), "blink should only be available when color is True"
        # Get prefix format according to color.
        error_prefix = self._get_prefix("ERROR", color, blink=True)
        warn_prefix = self._get_prefix("WARNING", color, blink=True)
        info_prefix = self._get_prefix("INFO", color, blink)
        debug_prefix = self._get_prefix("DEBUG", color, blink)

        # Config output format.
        self.err_format = (
            f"%(asctime)s - %(name)s - {error_prefix} - "
            "%(pathname)s - %(funcName)s - %(lineno)d - "
            "%(message)s"
        )
        self.warn_format = f"%(asctime)s - %(name)s - {warn_prefix} - %(" "message)s"
        self.info_format = f"%(asctime)s - %(name)s - {info_prefix} - %(" "message)s"
        self.debug_format = f"%(asctime)s - %(name)s - {debug_prefix} - %(" "message)s"

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ["underline"]
            if blink:
                attrs.append("blink")
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class ABLLogger(Logger, ManagerMixin):
    """Formatted logger used to record messages.

    ``ABLLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``ABLLogger`` has the following features:

    - Distributed log storage, ``ABLLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``ABLLogger`` could
          be different. We can only get ``ABLLogger`` instance by
          ``ABLLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``ABLLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``ABLLogger`` will not log warning
          or error message without ``Handler``.

    Examples:
        >>> logger = ABLLogger.get_instance(name='ABLLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'ABLLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = ABLLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = ABLLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = ABLLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'abl'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
    """

    def __init__(
        self,
        name: str,
        logger_name="abl",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w"
    ):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(ABLFormatter(color=True, datefmt="%m/%d %H:%M:%S"))
        stream_handler.setLevel(log_level)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is None:
            import time
            local_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())

            save_dir = os.path.join("results", local_time)
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            log_file = osp.join(save_dir, local_time + ".log")

        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(
            ABLFormatter(color=False, datefmt="%Y/%m/%d %H:%M:%S")
        )
        file_handler.setLevel(log_level)
        file_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> "ABLLogger":
        """Get latest created ``ABLLogger`` instance.

        :obj:`ABLLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "abl".

        Returns:
            ABLLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance("abl")
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``ABLLogger`` is not managed by ``logging.Manager`` anymore,
        ``ABLLogger`` should override this method to clear caches of all
        ``ABLLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in ABLLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(
    msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO
) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = ABLLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if ABLLogger.check_instance_created(logger):
            logger_instance = ABLLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"ABLLogger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )
