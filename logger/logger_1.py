import logging

# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')

'''By using the level parameter, you can set what level of log messages you want to record.
This can be done by passing one of the constants available in the class, and 
this would enable all logging calls at or above
that level to be logged'''
logging.basicConfig(level=logging.DEBUG)
logging.debug('This is a debug message')

'''Similarly, for logging to a file rather than the console, filename and 
filemode can be used, and you can decide 
the format of the message using format.'''
logging.basicConfig(filename='app.log',filemode='w',format='%(name)s-%(levelname)s-%(messae)s')
logging.warning('This will get logged to a file')