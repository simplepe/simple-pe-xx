import logging
import sys
from pesummary.utils.utils import _logger_format

stream_level = 'INFO'
if "-v" in sys.argv or "--verbose" in sys.argv:
    stream_level = 'DEBUG'

logging.basicConfig(format=_logger_format(), datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger('simple-pe')
logger.setLevel(getattr(logging, stream_level))
