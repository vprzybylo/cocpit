def auto_str(cls):
    """
    - Automatically implements a string representation for classes
      instead of memory id
    - Finds all attributes of the class and prints them in a readable format
    - Called with str(object) or str(instance) after using @auto_str decorator
      above class name
    """

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%s" % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls
