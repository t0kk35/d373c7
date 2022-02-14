"""
Common classes for all features
(c) 2020 d373c7
"""
import logging


class EngineContextException(Exception):
    def __init__(self, message: str):
        super().__init__('Error creating Engine context; ' + message)


class EngineContext:
    """ Base class for engine creation. Implemented a context for future use. All engines will be implemented a context
    in order to be able to provide data and create/keep/destroy connections and resources.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s.%(msecs)03d %(name)-30s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(__name__)
        logger.info('Start Engine...')

    def __enter__(self) -> 'EngineContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
