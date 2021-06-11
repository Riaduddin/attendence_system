import logging
import employ
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler=logging.FileHandler('sample.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# logging.basicConfig(filename='sample.log', level=logging.DEBUG,
#                     format='%(asctime)s:%(levelname)s:%(message)s')


def add(x, y):
    """Add Function"""
    return x + y


def subtract(x, y):
    """Subtract Function"""
    return x - y


def multiply(x, y):
    """Multiply Function"""
    return x * y


def divide(x, y):
    """Divide Function"""
    try:
        return x/y
    except ZeroDivisionError:
        logging.exception('Tried to divide by zero')
    else:
        return result


num_1 = 20
num_2 = 0

add_result = add(num_1, num_2)
logger.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

sub_result = subtract(num_1, num_2)
logger.debug('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))

mul_result = multiply(num_1, num_2)
logger.debug('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))

div_result = divide(num_1, num_2)
logger.debug('Div: {} / {} = {}'.format(num_1, num_2, div_result))