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
    """
    Filter for eliminating repeated warning messages in logging.

    This filter checks for duplicate warning messages and allows only the first occurrence of
    each message to be logged, filtering out subsequent duplicates.

    Parameters
    ----------
    name : str, optional
        The name of the filter, by default "abl".
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
    """
    Colorful format for ABLLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Parameters
    ----------
    color : bool
        Whether to use colorful format. filehandler is not
        allowed to use color format, otherwise it will be garbled.
    blink : bool
        Whether to blink the ``INFO`` and ``DEBUG`` logging
        level.
    kwargs : dict
        Keyword arguments passed to
        :meth:`logging.Formatter.__init__`.
    """

    _color_mapping: dict = dict(ERROR="red", WARNING="yellow", INFO="white", DEBUG="green")

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (not color and blink), "blink should only be available when color is True"
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
        """
        Get the prefix of the target log level.

        Parameters
        ----------
        level : str
            Log level.
        color : bool
            Whether to get a colorful prefix.
        blink : bool, optional
            Whether the prefix will blink.

        Returns
        -------
        str
            The plain or colorful prefix.
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
        """
        Override the ``logging.Formatter.format`` method. Output the
        message according to the specified log level.

        Parameters
        ----------
        record : LogRecord
            A LogRecord instance representing an event being logged.

        Returns
        -------
        str
            Formatted result.
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
    """
    Formatted logger used to record messages with different log levels and features.

    `ABLLogger` provides a formatted logger that can log messages with different
    log levels. It allows the creation of logger instances in a similar manner to `ManagerMixin`.
    The logger has features like distributed log storage and colored terminal output for different
    log levels.

    Parameters
    ----------
    name : str
        Global instance name.
    logger_name : str, optional
        `name` attribute of `logging.Logger` instance. Defaults to 'abl'.
    log_file : str, optional
        The log filename. If specified, a `FileHandler` will be added to the logger.
        Defaults to None.
    log_level : Union[int, str]
        The log level of the handler. Defaults to 'INFO'.
        If log level is 'DEBUG', distributed logs will be saved during distributed training.
    file_mode : str
        The file mode used to open log file. Defaults to 'w'.

    Notes
    -----
    - The `name` of the logger and the `instance_name` of `ABLLogger` could be different.
      `ABLLogger` instances are retrieved using `ABLLogger.get_instance`, not `logging.getLogger`.
      This ensures `ABLLogger` is not influenced by third-party logging configurations.
    - Unlike `logging.Logger`, `ABLLogger` will not log warning or error messages without `Handler`.

    Examples
    --------
    >>> logger = ABLLogger.get_instance(name='ABLLogger', logger_name='Logger')
    >>> # Although logger has a name attribute like `logging.Logger`
    >>> # We cannot get logger instance by `logging.getLogger`.
    >>> assert logger.name == 'Logger'
    >>> assert logger.instance_name == 'ABLLogger'
    >>> assert id(logger) != id(logging.getLogger('Logger'))
    >>> # Get logger that does not store logs.
    >>> logger1 = ABLLogger.get_instance('logger1')
    >>> # Get logger only save rank0 logs.
    >>> logger2 = ABLLogger.get_instance('logger2', log_file='out.log')
    >>> # Get logger only save multiple ranks logs.
    >>> logger3 = ABLLogger.get_instance('logger3', log_file='out.log', distributed=True)
    """

    def __init__(
        self,
        name: str,
        logger_name="abl",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w",
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

            _log_dir = os.path.join("results", local_time)
            self._log_dir = _log_dir
            if not os.path.exists(_log_dir):
                os.makedirs(_log_dir)
            log_file = osp.join(_log_dir, local_time + ".log")

        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(ABLFormatter(color=False, datefmt="%Y/%m/%d %H:%M:%S"))
        file_handler.setLevel(log_level)
        file_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @property
    def log_dir(self):
        return self._log_dir

    @classmethod
    def get_current_instance(cls) -> "ABLLogger":
        """
        Get the latest created `ABLLogger` instance.

        Returns
        -------
        ABLLogger
            The latest created `ABLLogger` instance. If no instance has been created,
            returns a logger with the instance name "abl".
        """
        if not cls._instance_dict:
            cls.get_instance("abl")
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """
        Pass a record to all relevant handlers.

        Override the `callHandlers` method in `logging.Logger` to avoid
        multiple warning messages in DDP mode. This method loops through all
        handlers of the logger instance and its parents in the logger hierarchy.

        Parameters
        ----------
        record : LogRecord
            A `LogRecord` instance containing the logged message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """
        Set the logging level of this logger.

        Override the `setLevel` method to clear caches of all `ABLLogger` instances
        managed by `ManagerMixin`. The level must be an int or a str.

        Parameters
        ----------
        level : Union[int, str]
            The logging level to set.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in ABLLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO) -> None:
    """
    Print a log message using the specified logger or a default method.

    This function logs a message with a given logger, if provided, or prints it using
    the standard `print` function. It supports special logger types such as 'silent' and 'current'.

    Parameters
    ----------
    msg : str
        The message to be logged.
    logger : Optional[Union[Logger, str]], optional
        The logger to use for logging the message. It can be a `logging.Logger` instance, a string
        specifying the logger name, 'silent', 'current', or None. If None, the `print`
        method is used.
        - 'silent': No message will be printed.
        - 'current': Use the latest created logger to log the message.
        - other str: The instance name of the logger. A `ValueError` is raised if the logger has not
        been created.
        - None: The `print()` method is used for logging.
    level : int, optional
        The logging level. This is only applicable when `logger` is a Logger object, 'current',
        or a named logger instance. The default is `logging.INFO`.
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
