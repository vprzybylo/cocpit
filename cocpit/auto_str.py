def auto_str(cls):
    '''
    automatically implements a string representation for classes
    instead of memory id
    finds all attributes of the class
    called with str(instance)
    '''

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls
